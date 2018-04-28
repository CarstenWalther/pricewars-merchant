from abc import ABCMeta, abstractmethod
import time
import threading
import hashlib
import base64
from typing import Optional, List

from api import Marketplace, Producer
from server import MerchantServer
from models import SoldOffer, Offer


class PricewarsMerchant(metaclass=ABCMeta):

    def __init__(self, port: int, token: Optional[str], marketplace_url: str, producer_url: str, name: str):
        self.settings = {
            'update interval': 4,
            'restock limit': 20,
            'order threshold': 0,
            'shipping': 5,
            'primeShipping': 1,
        }
        self.state = 'running'
        self.server_thread = self.start_server(port)
        self.pending_order = None

        self.marketplace = Marketplace(token, host=marketplace_url)
        self.marketplace.wait_for_host()

        if token:
            self.token = token
            self.merchant_id = self.calculate_id(token)
        else:
            register_response = self.marketplace.register(port, name)
            self.token = register_response.merchant_token
            self.merchant_id = register_response.merchant_id

        self.producer = Producer(self.token, host=producer_url)

    @staticmethod
    def calculate_id(token: str) -> str:
        return base64.b64encode(hashlib.sha256(token.encode('utf-8')).digest()).decode('utf-8')

    def run(self):
        start_time = time.time()
        while True:
            if self.state == 'running':
                self.update_offers()
            # Waiting for the length of the update interval minus the execution time
            time.sleep(self.settings['update interval'] -
                       ((time.time() - start_time) % self.settings['update interval']))

    def update_offers(self) -> None:
        """
        Entry point for regular merchant activity.
        When the merchant is running, this is called in each update interval.
        """
        market_situation = self.marketplace.get_offers()
        own_offers = [offer for offer in market_situation if offer.merchant_id == self.merchant_id]

        inventory_level = sum(offer.amount for offer in own_offers)
        product = None
        if self.pending_order is None and inventory_level <= self.settings['order threshold']:
            order_id = self.producer.order(self.settings['restock limit'] - inventory_level).id
            self.pending_order = order_id
        elif self.pending_order is not None:
            order = self.producer.receive_items(self.pending_order)
            product = order.product
            #self.restock(self.pending_order, market_situation)
            self.pending_order = None

        if own_offers:
            # This merchant has at most one active offer
            offer = own_offers[0]
            new_price = self.calculate_price(offer.offer_id, market_situation)
            print('Competitor update price to', new_price)
            offer.price = new_price
            self.marketplace.update_offer(offer)
            if product:
                self.marketplace.restock(offer.offer_id, product.amount, product.signature)
        elif product:
            shipping_time = {
                'standard': self.settings['shipping'],
                'prime': self.settings['primeShipping']
            }
            offer = Offer.from_product(product, 0, shipping_time)
            offer.merchant_id = self.merchant_id
            offer.price = self.calculate_price(offer.offer_id, market_situation + [offer])
            print('Competitor update price to', offer.price)
            self.marketplace.add_offer(offer)

    def restock(self, order_id, market_situation):
        order = self.producer.receive_items(order_id)
        product = order.product

        offer = Offer.from_product(product, 0, shipping_time)
        offer.merchant_id = self.merchant_id
        offer.price = self.calculate_price(offer.offer_id, market_situation + [offer])
        self.marketplace.add_offer(offer)

    def sold_offer(self, offer: SoldOffer) -> None:
        """
        This method is called whenever the merchant sells a product.
        """
        print('Product sold')

    def start(self):
        self.state = 'running'

    def stop(self):
        self.state = 'stopping'

    def update_settings(self, new_settings: dict) -> None:
        for key, value in new_settings.items():
            if key in self.settings:
                # Cast value type to the type that is already in the settings dictionary
                value = type(self.settings[key])(value)
            self.settings[key] = value

    def start_server(self, port):
        server = MerchantServer(self)
        thread = threading.Thread(target=server.app.run, kwargs={'host': '0.0.0.0', 'port': port})
        thread.daemon = True
        thread.start()
        return thread

    @abstractmethod
    def calculate_price(self, offer_id: int, market_situation: List[Offer]) -> float:
        """
        Calculate the price for the offer indicated by 'offer_id' given the current market situation.
        The offer id is guaranteed to be in the market situation.
        """
        pass

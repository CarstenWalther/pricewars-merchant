import json
from abc import ABCMeta, abstractmethod
import time
import threading
import hashlib
import base64
import random
from typing import Optional, List

from api import Marketplace, Producer
from server import MerchantServer
from models import SoldOffer, Offer


class PricewarsMerchant(metaclass=ABCMeta):
    TOKEN_FILE = 'auth_tokens.json'

    def __init__(self, port: int, token: Optional[str], marketplace_url: str, producer_url: str, merchant_name: str):
        self.settings = {
            'update_interval': 5,
            # it could make sense to choose larger upper bounds to
            # ensure that the merchants to not exceed their quota.
            'interval_lower_bound_relative': 0.7,
            'interval_upper_bound_relative': 1.35,
            'restock_limit': 20,
            'shipping': 5,
            'prime_shipping': 1,
        }
        self.state = 'running'
        self.server_thread = self.start_server(port)
        self.number_offered_items = 0
        self.products_not_offered = []

        if not token:
            token = self.load_tokens().get(merchant_name)

        self.marketplace = Marketplace(token, host=marketplace_url)
        self.marketplace.wait_for_host()

        if token:
            merchant_id = self.calculate_id(token)
            if not self.marketplace.merchant_exists(merchant_id):
                print('Existing token appears to be outdated.')
                token = None
            else:
                print('Running with existing token "%s".' % token)
                self.token = token
                self.merchant_id = merchant_id

        if token is None:
            register_response = self.marketplace.register(port, merchant_name)
            self.token = register_response.merchant_token
            self.merchant_id = register_response.merchant_id
            self.save_token(merchant_name)
            print('Registered new merchant with token "%s".' % self.token)

        # request current request limitations from market place.
        req_limit = self.marketplace.get_request_limit()

        # Update rate has to account of (i) getting market situations,
        # (ii) posting updates, (iii) getting products, (iv) posting
        # new products. As restocking should not occur too often,
        # we use a rather conservative factor of 2.5x factor.
        self.settings['update_interval'] = (1 / req_limit) * 2.5

        self.producer = Producer(self.token, host=producer_url)

    def load_tokens(self) -> dict:
        try:
            with open(self.TOKEN_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_token(self, name: str) -> None:
        tokens = self.load_tokens()
        with open(self.TOKEN_FILE, 'w') as f:
            tokens[name] = self.token
            json.dump(tokens, f)

    @staticmethod
    def calculate_id(token: str) -> str:
        return base64.b64encode(hashlib.sha256(token.encode('utf-8')).digest()).decode('utf-8')

    def run(self):
        # initial random sleep to avoid starting merchants in sync
        time.sleep(2 * random.random())

        start_time = time.time()
        update_counter = 1
        self.open_new_offer()
        while True:
            interval = self.settings['update_interval']
            lower_bound = self.settings['interval_lower_bound_relative']
            upper_bound = self.settings['interval_upper_bound_relative']

            if self.state == 'running':
                self.update_offers()

            # determine required sleep length for next interval
            rdm_interval_length = random.uniform(interval * lower_bound, interval * upper_bound)
            # calculate next expected update timestamp (might be in the
            # past in cases where the marketplace blocked for some time)
            next_update_ts = start_time + interval * (update_counter - 1) + rdm_interval_length
            sleep_time = next_update_ts - time.time()

            if sleep_time <= 0:
                # short random sleep to catch up with the intervals,
                # but try not to DDoS the marketplace
                sleep_time = random.uniform(interval * 0.05, interval * 0.2)

            time.sleep(sleep_time)
            update_counter += 1

    def update_offers(self) -> None:
        """
        Entry point for regular merchant activity.
        When the merchant is running, this is called in each update interval.
        """
        market_situation = self.marketplace.get_offers()
        own_offers = [offer for offer in market_situation if offer.merchant_id == self.merchant_id]

        for offer in own_offers:
            offer.price = self.calculate_price(offer.offer_id, market_situation)
            self.marketplace.update_offer(offer)

    def restock(self):
        order = self.producer.order(self.settings['restock_limit'])
        return order.products

    def open_new_offer(self) -> None:
        if not self.products_not_offered:
            self.products_not_offered = self.restock()

        product = self.products_not_offered.pop()
        self.number_offered_items += product.quantity
        shipping_time = {
            'standard': self.settings['shipping'],
            'prime': self.settings['prime_shipping']
        }
        offer = Offer.from_product(product, 0, shipping_time)
        offer.merchant_id = self.merchant_id
        market_situation = self.marketplace.get_offers()
        offer.price = self.calculate_price(offer.offer_id, market_situation + [offer])
        self.marketplace.add_offer(offer)

    def sold_offer(self, offer: SoldOffer) -> None:
        """
        This method is called whenever the merchant sells a product.
        """
        print('Product sold')
        self.number_offered_items -= offer.quantity_sold
        if self.number_offered_items == 0:
            self.open_new_offer()

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

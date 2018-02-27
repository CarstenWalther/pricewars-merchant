import argparse
import threading
import time
import numpy as np

from pricewars import MerchantServer
from pricewars.api import Marketplace, Producer, Kafka
from pricewars.models import SoldOffer
from pricewars.models import Offer
from policy.policy import create_policy
from policy.demand_learning import blablabla


class DynProgrammingMerchant:
    def __init__(self, name: str, port: int, marketplace_url: str, producer_url: str):
        self.server_thread = self.start_server(port)

        self.marketplace = Marketplace(host=marketplace_url)
        self.marketplace.wait_for_host()
        response = self.marketplace.register(port, name, algorithm_name='strategy calculated with dynamic programming')
        self.merchant_id = response.merchant_id
        self.token = response.merchant_token
        self.marketplace = Marketplace(self.token, marketplace_url)
        self.producer = Producer(self.token, producer_url)
        self.kafka_reverse_proxy = Kafka(self.token)

        self.MAX_STOCK = 40
        self.INTERVAL_LENGTH_IN_SECONDS = 1
        self.MINUTES_BETWEEN_TRAININGS = 1
        self.selling_price_low = 25
        self.selling_price_high = 35

        # only one product can be bought in this scenario
        product_info = self.producer.get_products()[0]
        self.fixed_order_cost = product_info.fixed_order_cost
        self.product_cost = product_info.price
        holding_cost_per_unit_per_minute = self.marketplace.holding_cost_rate()
        self.holding_cost_per_interval = self.INTERVAL_LENGTH_IN_SECONDS * (holding_cost_per_unit_per_minute / 60)

        self.demand_function = None
        self.next_training = time.time() + self.MINUTES_BETWEEN_TRAININGS * 60

        self.shipping_time = {
            'standard': 5,
            'prime': 1
        }

    def start_server(self, port):
        server = MerchantServer(self)
        thread = threading.Thread(target=server.app.run, kwargs={'host': '0.0.0.0', 'port': port})
        thread.daemon = True
        thread.start()
        return thread

    def update_offers(self):
        market_situation = self.marketplace.get_offers()
        own_offers = [offer for offer in market_situation if offer.merchant_id == self.merchant_id]

        # TODO: create policy if there is no own order
        if self.demand_function and own_offers:
            # Assume we have only one active offer
            own_offer_id = own_offers[0].offer_id
            start = time.time()
            order_policy_array, pricing_policy_array = create_policy(self.demand_function, self.product_cost,
                                                                     self.fixed_order_cost,
                                                                     self.holding_cost_per_interval, self.MAX_STOCK, market_situation, own_offer_id)
            print('Updating policy took', time.time() - start, 'seconds')
            order_policy = lambda stock: order_policy_array[np.clip(stock, 0, len(order_policy_array) - 1)]
            pricing_policy = lambda stock: pricing_policy_array[np.clip(stock, 0, len(pricing_policy_array) - 1)]
        else:
            print('Use default policy')
            order_policy = lambda stock: 10 if stock == 0 else 0
            pricing_policy = lambda stock: np.random.randint(self.selling_price_low, self.selling_price_high + 1)

        inventory_level = sum(offer.amount for offer in own_offers)
        # Convert because json module cannot serialize numpy numbers
        order_quantity = int(order_policy(inventory_level))
        # Convert because json module cannot serialize numpy numbers
        new_price = float(pricing_policy(inventory_level))
        print('Update price to', new_price)

        if order_quantity > 0:
            print('Order', order_quantity, 'units')
            order = self.producer.order(order_quantity)
            product = order.product

            if own_offers:
                own_offers[0].price = new_price
                self.marketplace.update_offer(own_offers[0])
                # There should be only one own offer
                self.marketplace.restock(own_offers[0].offer_id, product.amount, product.signature)
            else:
                offer = Offer.from_product(product, new_price, self.shipping_time)
                self.marketplace.add_offer(offer)
        elif own_offers:
            # There should be only one own offer
            own_offers[0].price = new_price
            self.marketplace.update_offer(own_offers[0])

    def run(self):
        start_time = time.time()
        while True:
            self.update_offers()
            if time.time() >= self.next_training:
                threading.Thread(target=self.estimate_demand_distribution).start()
                self.next_training += self.MINUTES_BETWEEN_TRAININGS * 60
            time.sleep(self.INTERVAL_LENGTH_IN_SECONDS - ((time.time() - start_time) % self.INTERVAL_LENGTH_IN_SECONDS))

    def estimate_demand_distribution(self):
        start = time.time()
        market_situations = self.kafka_reverse_proxy.download_topic_data('marketSituation')
        sales_data = self.kafka_reverse_proxy.download_topic_data('buyOffer')
        demand_function = blablabla(market_situations, sales_data, self.merchant_id, self.INTERVAL_LENGTH_IN_SECONDS)
        if demand_function:
            self.demand_function = demand_function
        else:
            print('Failed to estimate demand. Use previous demand estimation')
        print('Learning took', time.time() - start, 'seconds')

    def sold_offer(self, offer: SoldOffer):
        print('Sold', offer.amount_sold, 'item(s)')

    def get_state(self) -> str:
        return 'running'

    def get_settings(self) -> dict:
        return {}

    def update_settings(self, new_settings):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description='PriceWars Merchant Being Cheapest')
    parser.add_argument('--name', type=str, default='Scenario 4 Merchant', help='Merchant name')
    parser.add_argument('--port', type=int, required=True, help='port to bind flask App to')
    parser.add_argument('--marketplace', type=str, default=Marketplace.DEFAULT_URL, help='Marketplace URL')
    parser.add_argument('--producer', type=str, default=Producer.DEFAULT_URL, help='Producer URL')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    merchant = DynProgrammingMerchant(args.name, args.port, args.marketplace, args.producer)
    merchant.run()

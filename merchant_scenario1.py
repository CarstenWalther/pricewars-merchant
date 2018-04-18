import argparse
import threading
import time
import numpy as np

from server import MerchantServer
from api import Marketplace, Producer
from models import SoldOffer
from models import Offer
from policy.policy_monopoly import create_policies

"""
This merchant operates in the following scenario:
- costumer demand distribution is known
- selling price is fixed
- monopoly
The merchant uses dynamic programming to calculate optimal ordering policies
"""

# the distribution of the customer arrival times is known
def time_between_visits():
    mean_customers_per_second = 100 / 60
    return np.random.exponential(scale= 1/mean_customers_per_second)


def estimate_demand_distribution(interval_length_in_seconds, array_length, iterations=10000):
    event_counter = np.zeros(array_length)
    interval_time = 0.0
    event_count = 0
    for _ in range(iterations):
        interval_time += time_between_visits()
        while interval_time >= interval_length_in_seconds:
            event_counter[min(event_count, array_length-1)] += 1
            event_count = 0
            interval_time -= interval_length_in_seconds
        event_count += 1

    return event_counter / np.sum(event_counter)


class DynProgrammingMerchant:
    def __init__(self, port: int, marketplace_url: str, producer_url: str):
        self.server_thread = self.setup_server(port)

        self.marketplace = Marketplace(marketplace_url)
        self.marketplace.wait_for_host()
        response = self.marketplace.register(endpoint_url_or_port=port, merchant_name='DynProgramming',
                                             algorithm_name='strategy calculated with dynamic programming')
        self.merchant_id = response.merchant_id
        self.token = response.merchant_token
        print('token', self.token)
        self.producer = Producer(self.token, producer_url)

        self.inventory = 0
        MAX_STOCK = 40
        self.INTERVAL_LENGTH_IN_SECONDS = 1.0
        # fix selling price
        self.SELLING_PRICE = 30

        # only one product can be bought in this scenario
        product_info = self.producer.get_products()[0]
        fixed_order_cost = product_info.fixed_order_cost
        product_cost = product_info.price
        holding_cost_per_unit_per_minute = self.marketplace.holding_cost_rate(self.merchant_id)
        holding_cost_per_interval = self.INTERVAL_LENGTH_IN_SECONDS * (holding_cost_per_unit_per_minute / 60)

        demand_distribution = estimate_demand_distribution(self.INTERVAL_LENGTH_IN_SECONDS, MAX_STOCK + 1)

        def distribution_function(demand, price):
            return demand_distribution[demand]

        # The pricing policy will be self.SELLING_PRICE for all decisions, because we reduce the price space to this one price.
        order_policy, pricing_policy = create_policies(distribution_function, product_cost, fixed_order_cost,
                                     holding_cost_per_interval, MAX_STOCK, self.SELLING_PRICE, self.SELLING_PRICE)

        self.order_policy = order_policy
        self.pricing_policy = pricing_policy

        self.shipping_time = {
            'standard': 5,
            'prime': 1
        }

    def setup_server(self, port):
        server = MerchantServer(self)
        thread = threading.Thread(target=server.app.run, kwargs={'host': '0.0.0.0', 'port': port})
        thread.daemon = True
        thread.start()
        return thread

    def update(self):
        order_quantity = self.order_policy(self.inventory)
        if order_quantity > 0:
            print('order', order_quantity, 'units')
            order = self.producer.order(order_quantity)
            product = order.product
            offer = Offer.from_product(product, self.pricing_policy(self.inventory), self.shipping_time)
            offer = self.marketplace.add_offer(offer)
            self.inventory += offer.amount

    def run(self):
        while True:
            self.update()
            time.sleep(self.INTERVAL_LENGTH_IN_SECONDS)

    def sold_offer(self, offer: SoldOffer):
        self.inventory -= offer.amount_sold
        print('bought', offer.amount_sold, 'available', self.inventory)

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
    parser.add_argument('--port', type=int, required=True, help='port to bind flask App to')
    parser.add_argument('--marketplace', type=str, default=Marketplace.DEFAULT_URL, help='Marketplace URL')
    parser.add_argument('--producer', type=str, default=Producer.DEFAULT_URL, help='Producer URL')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    merchant = DynProgrammingMerchant(args.port, args.marketplace, args.producer)
    merchant.run()

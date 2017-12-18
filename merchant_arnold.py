import argparse
import threading
import time
import requests
import numpy as np

from pricewars import MerchantServer
from pricewars.api import Marketplace, Producer
from pricewars.models import SoldOffer
from pricewars.models import Offer
from demand_estimation import estimate_demand_distribution


class ArnoldMerchant:
    def __init__(self, port: int, marketplace_url: str, producer_url: str):
        self.server_thread = self.setup_server(port)

        self.marketplace = Marketplace(marketplace_url)
        self.marketplace.wait_for_host()
        response = self.marketplace.register(endpoint_url_or_port=port, merchant_name='Arnold',
                                             algorithm_name='economic order quantity')
        self.merchant_id = response.merchant_id
        self.token = response.merchant_token
        self.producer = Producer(self.token, producer_url)

        self.inventory = 0
        self.MAX_STOCK = 40
        # fix selling price
        self.selling_price = 30

        # customer behavior is a known distribution
        def time_between_visits():
            return np.random.exponential(scale=1.0)

        # only one product can be bought in this scenario
        product_info = self.producer.get_products()[0]
        self.fixed_order_cost = product_info.fixed_order_cost
        self.product_cost = product_info.price
        # merchant should not be able to change his inventory price
        # but for convenience set the price here
        requests.put('http://marketplace:8080/holding_cost_rate', json={'rate': 5, 'merchant_id': self.merchant_id})
        self.holding_cost_per_unit_per_minute = self.marketplace.inventory_price()
        self.INTERVAL_LENGTH_IN_SECONDS = 1.0

        demand_distribution = estimate_demand_distribution(time_between_visits, self.INTERVAL_LENGTH_IN_SECONDS, self.MAX_STOCK + 1)

        def distribution_function(demand):
            return demand_distribution[demand]

        self.policy = self.create_policy(distribution_function)

        self.shipping_time = {
            'standard': 5,
            'prime': 1
        }

    def order_cost(self, order_quantity):
        return order_quantity * self.product_cost + (order_quantity > 0) * self.fixed_order_cost

    def holding_cost(self, remaining_stock, order_quantity):
        return (remaining_stock + order_quantity) * (self.holding_cost_per_unit_per_minute * self.INTERVAL_LENGTH_IN_SECONDS / 60)

    def sales_revenue(self, sales):
        return sales * self.selling_price

    def profit(self, remaining_stock, sales, order_quantity):
        return self.sales_revenue(sales) - self.order_cost(order_quantity) - self.holding_cost(remaining_stock, order_quantity)

    def create_policy(self, demand_distribution):
        policy = np.zeros(self.MAX_STOCK + 1, dtype=np.int32)
        expected_profits = np.zeros(self.MAX_STOCK + 1)

        for iteration in range(1000):
            remaining_stock, order_quantity, demand = np.split(
                np.mgrid[0:self.MAX_STOCK + 1, 0:self.MAX_STOCK + 1, 0:self.MAX_STOCK + 1], 3)
            remaining_stock = np.squeeze(remaining_stock)
            order_quantity = np.squeeze(order_quantity)
            demand = np.squeeze(demand)
            probabilities = demand_distribution(demand)
            sales = np.minimum(demand, remaining_stock + order_quantity)
            all_expected_profits = np.sum(probabilities * (
                    self.profit(remaining_stock, sales, order_quantity)
                    + iteration * expected_profits[np.clip(remaining_stock + order_quantity - sales, 0, self.MAX_STOCK)]
            ), axis=-1) / (iteration + 1)
            policy = np.argmax(all_expected_profits, axis=-1)
            expected_profits = np.max(all_expected_profits, axis=-1)
        print(policy)
        print('expected profit', expected_profits[0])

        def policy_function(remaining_stock):
            return policy[np.clip(remaining_stock, 0, self.MAX_STOCK)]

        return policy_function

    def setup_server(self, port):
        server = MerchantServer(self)
        thread = threading.Thread(target=server.app.run, kwargs={'host': '0.0.0.0', 'port': port})
        thread.daemon = True
        thread.start()
        return thread

    def update(self):
        order_quantity = self.policy(self.inventory)
        if order_quantity > 0:
            print('order', order_quantity, 'units')
            order = self.producer.order(order_quantity)
            product = order.product
            offer = Offer.from_product(product, self.selling_price, self.shipping_time)
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
    merchant = ArnoldMerchant(args.port, args.marketplace, args.producer)
    merchant.run()

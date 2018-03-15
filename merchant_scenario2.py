import argparse
import threading
import time
import numpy as np
from sklearn import linear_model
from scipy.stats import poisson

from server import MerchantServer
from api import Marketplace, Producer, Kafka
from models import SoldOffer, Offer
from policy.order_policy import create_policy


def learning(X, y, fix_selling_price):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    mean = model.predict(fix_selling_price).squeeze()
    print('mean sales at price', fix_selling_price, 'is', mean)

    def distribution(demand):
        return poisson.pmf(demand, mean)
    return distribution


def aggregate_sales(sales_data, interval, default_price):
    return sales_data.set_index('timestamp').resample(str(interval) + 's')\
        .agg({'amount': 'sum', 'price': 'min'})\
        .rename(columns={'amount': 'sales', 'price': 'min_price'})\
        .fillna({'sales': 0, 'min_price': default_price})


def default_policy(remaining_stock):
    if remaining_stock == 0:
        # Buy some products to avoid to miss sales
        return 10
    else:
        return 0


class DynProgrammingMerchant:
    def __init__(self, port: int, marketplace_url: str, producer_url: str):
        self.server_thread = self.start_server(port)

        self.marketplace = Marketplace(host=marketplace_url)
        self.marketplace.wait_for_host()
        response = self.marketplace.register(endpoint_url_or_port=port, merchant_name='Scenario 2 Merchant',
                                             algorithm_name='strategy calculated with dynamic programming')
        self.merchant_id = response.merchant_id
        self.token = response.merchant_token
        self.marketplace = Marketplace(self.token, marketplace_url)
        self.producer = Producer(self.token, producer_url)
        self.kafka_reverse_proxy = Kafka(self.token)

        self.inventory = 0
        self.MAX_STOCK = 40
        self.SELLING_PRICE = 30
        self.INTERVAL_LENGTH_IN_SECONDS = 1
        self.MINUTES_BETWEEN_TRAININGS = 1

        # only one product can be bought in this scenario
        product_info = self.producer.get_products()[0]
        self.fixed_order_cost = product_info.fixed_order_cost
        self.product_cost = product_info.price
        holding_cost_per_unit_per_minute = self.marketplace.holding_cost_rate()
        self.holding_cost_per_interval = self.INTERVAL_LENGTH_IN_SECONDS * (holding_cost_per_unit_per_minute / 60)

        self.policy = default_policy
        self.next_training = time.time() + self.MINUTES_BETWEEN_TRAININGS * 60

        self.shipping_time = {
            'standard': 5,
            'prime': 1
        }

    def estimate_demand_distribution(self, default_mean=1):
        sales_data = self.kafka_reverse_proxy.download_topic_data('buyOffer')
        if sales_data is None:
            print('No sales data: use default demand distribution')
            return lambda demand: poisson.pmf(demand, default_mean)
        else:
            sales = aggregate_sales(sales_data, self.INTERVAL_LENGTH_IN_SECONDS, self.SELLING_PRICE)
            return learning(sales[['min_price']], sales[['sales']], self.SELLING_PRICE)

    def update_policy(self):
        print('Update policy')
        demand_function = self.estimate_demand_distribution()
        policy_array = create_policy(demand_function, self.SELLING_PRICE, self.product_cost, self.fixed_order_cost,
                                     self.holding_cost_per_interval, self.MAX_STOCK)

        print(policy_array)
        self.policy = lambda remaining_stock: policy_array[np.clip(remaining_stock, 0, self.MAX_STOCK)]

    def start_server(self, port):
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
            offer = Offer.from_product(product, self.SELLING_PRICE, self.shipping_time)
            offer = self.marketplace.add_offer(offer)
            self.inventory += offer.amount

    def run(self):
        start_time = time.time()
        while True:
            self.update()
            if time.time() >= self.next_training:
                threading.Thread(target=self.update_policy).start()
                self.next_training += self.MINUTES_BETWEEN_TRAININGS * 60
            time.sleep(self.INTERVAL_LENGTH_IN_SECONDS - ((time.time() - start_time) % self.INTERVAL_LENGTH_IN_SECONDS))

    def sold_offer(self, offer: SoldOffer):
        self.inventory -= offer.amount_sold
        print('sold', offer.amount_sold, 'available', self.inventory)

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

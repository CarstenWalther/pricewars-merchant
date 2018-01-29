import argparse
import threading
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy.stats import poisson

from pricewars import MerchantServer
from pricewars.api import Marketplace, Producer, Kafka
from pricewars.models import SoldOffer
from pricewars.models import Offer
from policy.order_policy import create_policy


def demand_learning(X, y, fix_selling_price):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    mean = model.predict(fix_selling_price).squeeze()
    print('mean sales at price', fix_selling_price, 'is', mean)

    def demand_distribution(demand):
        return poisson.pmf(demand, mean)

    return demand_distribution


def default_policy(remaining_stock):
    # Refill the inventory with some products to avoid to miss sales
    return 10 if remaining_stock == 0 else 0


def aggregate_sales_to_market_situations(sales_data, market_situations):
    """
    This function sums up all sales that happen between each two successive market situations.
    The results are divided by the time between the two market situations to make
    them independent from the interval length.
    Sales are counted for each offer separately.
    """
    grouped = sales_data.groupby(['offer_id', pd.cut(sales_data['timestamp'], market_situations['timestamp'].unique(), right=False)])
    sales_by_interval = grouped['amount'].sum()
    # Calculate the time span from the start and end of the interval
    time_spans = sales_by_interval.index.get_level_values('timestamp') \
        .map(lambda e: e.right - e.left).astype('timedelta64[ns]').values
    sales_per_minute = 1 / (time_spans / np.timedelta64(1, 'm')) * sales_by_interval
    return sales_per_minute


def extract_features(market_situation, own_offer_id):
    # TODO: maybe use index here
    own_offer = market_situation[market_situation['offer_id'] == own_offer_id].iloc[0]
    return (own_offer['price'], )


def aggregate_sales_data(merchant_id, market_situations, sales_data):
    """
    This function creates a pair of features and sales for each market situation and offer.
    It returns a dictionary with product ids as keys and lists of the mentioned pairs as values.
    """
    sales_per_minute = aggregate_sales_to_market_situations(sales_data, market_situations)
    # We want to look up values with the timestamp of a market situation.
    # Thus the interval index is transformed to a timestamp index.
    sales_per_minute.index = sales_per_minute.index.map(lambda e: (e[0], e[1].left))
    sales_data_by_product = defaultdict(list)

    # We look at each marketsituation (same timestamp) and separate market data for each product.
    for (product_id, timestamp), market_situation in market_situations.groupby(['product_id', 'timestamp']):
        # A market situation can have multiple offers that belong to this merchant.
        # For each own offer a feature-sales-pair is generated that
        # assumes that all other offers are competitors offers.
        for own_offer_id in market_situation[market_situation["merchant_id"] == merchant_id]['offer_id']:
            features = extract_features(market_situation, own_offer_id)
            sales = sales_per_minute.get((own_offer_id, timestamp), default=0)
            sales_data_by_product[product_id].append((features, sales))

    # We cannot tell the sales per minute for the last market situation.
    # That is why the last trainings pair is removed.
    # TODO: What if last trainings pair does not belong to last market situation? There must be a better way
    for product_id in sales_data_by_product:
        sales_data_by_product[product_id] = sales_data_by_product[product_id][:-1]

    return sales_data_by_product


class DynProgrammingMerchant:
    def __init__(self, port: int, marketplace_url: str, producer_url: str):
        self.server_thread = self.start_server(port)

        self.marketplace = Marketplace(host=marketplace_url)
        self.marketplace.wait_for_host()
        response = self.marketplace.register(endpoint_url_or_port=port, merchant_name='Scenario 3 Merchant',
                                             algorithm_name='strategy calculated with dynamic programming')
        self.merchant_id = response.merchant_id
        self.token = response.merchant_token
        self.marketplace = Marketplace(self.token, marketplace_url)
        self.producer = Producer(self.token, producer_url)
        self.kafka_reverse_proxy = Kafka(self.token)

        self.inventory = 0
        self.MAX_STOCK = 40
        self.INTERVAL_LENGTH_IN_SECONDS = 1
        self.MINUTES_BETWEEN_TRAININGS = 5
        self.selling_price_low = 28
        self.selling_price_high = 32
        self.selling_price_mean = np.mean([self.selling_price_low, self.selling_price_high])

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
        market_situations = self.kafka_reverse_proxy.download_topic_data('marketSituation')
        sales_data = self.kafka_reverse_proxy.download_topic_data('buyOffer')
        #TODO: used for debuggin, remove later
        market_situations.to_csv('market_situation.csv', index=False)
        sales_data.to_csv('sales.csv', index=False)
        if market_situations is not None and sales_data is not None:
            sales_per_product = aggregate_sales_data(self.merchant_id, market_situations, sales_data)
            # Currently there is only one product type
            if sales_per_product[1]:
                features, sales_per_minute = zip(*sales_per_product[1])
                sales_per_decision_interval = np.array(sales_per_minute) / 60 * self.INTERVAL_LENGTH_IN_SECONDS
                return demand_learning(features, sales_per_decision_interval, self.selling_price_mean)
        print('No market data available: use default demand distribution')
        return lambda demand: poisson.pmf(demand, default_mean)

    def random_selling_price(self):
        return np.random.randint(self.selling_price_low, self.selling_price_high)

    def update_policy(self):
        print('Update policy')
        demand_function = self.estimate_demand_distribution()
        policy_array = create_policy(demand_function, self.selling_price_mean, self.product_cost, self.fixed_order_cost,
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
            offer = Offer.from_product(product, self.random_selling_price(), self.shipping_time)
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

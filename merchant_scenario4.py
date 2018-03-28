import argparse
import threading
import time

from server import MerchantServer
from api import Marketplace, Producer, Kafka
from models import SoldOffer, Offer
from policy.policy import PolicyOptimizer
from policy.demand_learning import demand_learning


class DynProgrammingMerchant:
    def __init__(self, name: str, port: int, marketplace_url: str, producer_url: str):
        self.server_thread = self.start_server(port)

        self.marketplace = Marketplace(host=marketplace_url)
        self.marketplace.wait_for_host()
        response = self.marketplace.register(port, name, algorithm_name='Dynamic Programming')
        self.merchant_id = response.merchant_id
        self.token = response.merchant_token
        self.marketplace = Marketplace(self.token, marketplace_url)
        self.producer = Producer(self.token, producer_url)
        self.kafka_reverse_proxy = Kafka(self.token)

        self.UPDATE_INTERVAL_IN_SECONDS = 4
        self.MINUTES_BETWEEN_TRAININGS = 1

        # only one product can be bought in this scenario
        product_info = self.producer.get_products()[0]
        self.fixed_order_cost = product_info.fixed_order_cost
        self.product_cost = product_info.price
        holding_cost_per_unit_per_minute = self.marketplace.holding_cost_rate()
        self.holding_cost_per_interval = self.UPDATE_INTERVAL_IN_SECONDS * (holding_cost_per_unit_per_minute / 60)

        self.policy_optimizer = PolicyOptimizer()
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

    def create_policy(self, competitor_offers):
        start = time.time()
        dummy_offer = Offer()
        market_situation = [*competitor_offers, dummy_offer]
        order_policy, pricing_policy = \
            self.policy_optimizer.create_policies(self.demand_function, self.product_cost, self.fixed_order_cost,
                                                  self.holding_cost_per_interval, market_situation,
                                                  dummy_offer.offer_id)
        print('Creating policy took {0:.2f} seconds'.format(time.time() - start))
        return order_policy, pricing_policy

    def update_offers(self):
        market_situation = self.marketplace.get_offers()
        own_offers = [offer for offer in market_situation if offer.merchant_id == self.merchant_id]
        competitor_offers = [offer for offer in market_situation if offer.merchant_id != self.merchant_id]
        order_policy, pricing_policy = self.create_policy(competitor_offers)
        inventory_level = sum(offer.amount for offer in own_offers)
        # Convert because json module cannot serialize numpy numbers
        order_quantity = int(order_policy(inventory_level))
        # Convert because json module cannot serialize numpy numbers
        new_price = float(pricing_policy(inventory_level))
        print('Update price to', new_price)

        product = None
        if order_quantity > 0:
            print('Order', order_quantity, 'units')
            order = self.producer.order(order_quantity)
            product = order.product
            self.fixed_order_cost = order.fixed_cost
            self.product_cost = order.unit_price

        if own_offers:
            # This merchant has at most one active offer
            own_offers[0].price = new_price
            self.marketplace.update_offer(own_offers[0])
            if product:
                self.marketplace.restock(own_offers[0].offer_id, product.amount, product.signature)
        elif product:
            offer = Offer.from_product(product, new_price, self.shipping_time)
            self.marketplace.add_offer(offer)

    def run(self):
        start_time = time.time()
        while True:
            self.update_offers()
            if time.time() >= self.next_training:
                threading.Thread(target=self.estimate_demand_distribution).start()
                self.next_training += self.MINUTES_BETWEEN_TRAININGS * 60
            time.sleep(self.UPDATE_INTERVAL_IN_SECONDS - ((time.time() - start_time) % self.UPDATE_INTERVAL_IN_SECONDS))

    def estimate_demand_distribution(self):
        start = time.time()
        market_situations = self.kafka_reverse_proxy.download_topic_data('marketSituation')
        sales_data = self.kafka_reverse_proxy.download_topic_data('buyOffer')
        demand_function = demand_learning(market_situations, sales_data, self.merchant_id,
                                          self.UPDATE_INTERVAL_IN_SECONDS)
        if demand_function:
            self.demand_function = demand_function
        else:
            print('Failed to estimate demand. Use previous demand estimation')
        print('Learning took', time.time() - start, 'seconds')

    def sold_offer(self, offer: SoldOffer):
        print('Sold', offer.amount_sold, 'item(s)')

    @property
    def state(self):
        return 'running'

    @property
    def settings(self):
        return {}

    def update_settings(self, new_settings):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description='PriceWars Merchant')
    parser.add_argument('--name', type=str, default='Data-Driven Merchant', help='Merchant name')
    parser.add_argument('--port', type=int, required=True, help='port to bind flask App to')
    parser.add_argument('--marketplace', type=str, default=Marketplace.DEFAULT_URL, help='Marketplace URL')
    parser.add_argument('--producer', type=str, default=Producer.DEFAULT_URL, help='Producer URL')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    merchant = DynProgrammingMerchant(args.name, args.port, args.marketplace, args.producer)
    merchant.run()

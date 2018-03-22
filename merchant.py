import argparse
from typing import Optional
import random

from api import Marketplace, Producer
from pricewars_merchant import PricewarsMerchant


class CheapestStrategy:
    name = 'Cheapest'

    settings = {
        'price decrement': 0.05,
        'default price': 30
    }

    @staticmethod
    def calculate_price(merchant, offer_id, market_situation):
        product_id = [offer for offer in market_situation if offer.offer_id == offer_id][0].product_id
        relevant_competitor_offers = [offer for offer in market_situation if
                                      offer.product_id == product_id and
                                      offer.merchant_id != merchant.merchant_id]
        if not relevant_competitor_offers:
            return merchant.settings['default price']

        cheapest_offer = min(relevant_competitor_offers, key=lambda offer: offer.price)
        return cheapest_offer.price - merchant.settings['price decrement']


class TwoBoundStrategy:
    name = 'Two Bound'

    settings = {
        'price decrement': 0.10,
        'upper price bound': 30,
        'lower price bound': 20
    }

    @staticmethod
    def calculate_price(merchant, offer_id, market_situation):
        product_id = [offer for offer in market_situation if offer.offer_id == offer_id][0].product_id
        relevant_competitor_offers = [offer for offer in market_situation if
                                      offer.product_id == product_id and
                                      offer.merchant_id != merchant.merchant_id]
        if not relevant_competitor_offers:
            return merchant.settings['upper price bound']

        cheapest_offer = min(relevant_competitor_offers, key=lambda offer: offer.price)
        if cheapest_offer.price <= merchant.settings['lower price bound']:
            return merchant.settings['upper price bound']
        else:
            return cheapest_offer.price - merchant.settings['price decrement']

class TwoBoundStrategyA:
    name = 'Two Bound Merchant A'

    settings = {
        'price decrement': 0.20,
        'upper price bound': 35,
        'lower price bound': 15,
        'update interval': 10,
        'restock limit': 20,
        'order threshold': 5,
    }

    @staticmethod
    def calculate_price(merchant, offer_id, market_situation):
        product_id = [offer for offer in market_situation if offer.offer_id == offer_id][0].product_id
        relevant_competitor_offers = [offer for offer in market_situation if
                                      offer.product_id == product_id and
                                      offer.merchant_id != merchant.merchant_id]
        if not relevant_competitor_offers:
            return merchant.settings['upper price bound']

        cheapest_offer = min(relevant_competitor_offers, key=lambda offer: offer.price)
        if cheapest_offer.price <= merchant.settings['lower price bound']:
            return merchant.settings['upper price bound']
        else:
            return min(cheapest_offer.price - merchant.settings['price decrement'], merchant.settings['upper price bound'])

class TwoBoundStrategyB:
    name = 'Two Bound Merchant B'

    settings = {
        'price decrement': 0.20,
        'upper price bound': 30,
        'lower price bound': 10,
        'update interval': 4,
        'restock limit': 15,
        'order threshold': 0,
    }

    @staticmethod
    def calculate_price(merchant, offer_id, market_situation):
        product_id = [offer for offer in market_situation if offer.offer_id == offer_id][0].product_id
        relevant_competitor_offers = [offer for offer in market_situation if
                                      offer.product_id == product_id and
                                      offer.merchant_id != merchant.merchant_id]
        if not relevant_competitor_offers:
            return merchant.settings['upper price bound']

        cheapest_offer = min(relevant_competitor_offers, key=lambda offer: offer.price)
        if cheapest_offer.price <= merchant.settings['lower price bound']:
            return merchant.settings['upper price bound']
        else:
            return min(cheapest_offer.price - merchant.settings['price decrement'],
                       merchant.settings['upper price bound'])


class RandomStrategy:
    name = 'Random'

    settings = {}

    @staticmethod
    def calculate_price(merchant, offer_id, market_situation):
        return random.randint(20, 50)


class Merchant(PricewarsMerchant):
    def __init__(self, token: Optional[str], port: int, marketplace_url: str, producer_url: str, strategy):
        super().__init__(port, token, marketplace_url, producer_url, strategy.name)
        self.strategy = strategy
        self.settings.update(strategy.settings)

    def calculate_price(self, offer_id, market_situation):
        return self.strategy.calculate_price(self, offer_id, market_situation)


def parse_arguments():
    parser = argparse.ArgumentParser(description='PriceWars Merchant')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--port', type=int, help='port to bind flask App to')
    group.add_argument('--token', type=str, help='Merchant secret token')
    parser.add_argument('--marketplace', type=str, default=Marketplace.DEFAULT_URL, help='Marketplace URL')
    parser.add_argument('--producer', type=str, default=Producer.DEFAULT_URL, help='Producer URL')
    parser.add_argument('--strategy', type=str, required=True,
                        help="Chose the merchant's strategy (example: Cheapest, Two Bound)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    strategies = {
        CheapestStrategy.name: CheapestStrategy,
        TwoBoundStrategy.name: TwoBoundStrategy,
        TwoBoundStrategyA.name: TwoBoundStrategyA,
        TwoBoundStrategyB.name: TwoBoundStrategyB,
        RandomStrategy.name: RandomStrategy,
    }
    merchant = Merchant(args.token, args.port, args.marketplace, args.producer, strategies[args.strategy])
    merchant.run()


if __name__ == '__main__':
    main()

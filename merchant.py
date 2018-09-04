import argparse
from typing import Optional

from api import Marketplace, Producer
from pricewars_merchant import PricewarsMerchant


class CheapestStrategy:
    name = 'cheapest'

    settings = {
        'price_decrement': 0.05,
        'default_price': 30
    }

    @staticmethod
    def calculate_price(merchant, offer_id, market_situation):
        product_id = [offer for offer in market_situation if offer.offer_id == offer_id][0].product_id
        relevant_competitor_offers = [offer for offer in market_situation if
                                      offer.product_id == product_id and
                                      offer.merchant_id != merchant.merchant_id]
        if not relevant_competitor_offers:
            return merchant.settings['default_price']

        cheapest_offer = min(relevant_competitor_offers, key=lambda offer: offer.price)
        return cheapest_offer.price - merchant.settings['price_decrement']


class TwoBoundStrategy:
    name = 'two_bound'

    settings = {
        'price_decrement': 0.10,
        'upper_price_bound': 30,
        'lower_price_bound': 20
    }

    @staticmethod
    def calculate_price(merchant, offer_id, market_situation):
        product_id = [offer for offer in market_situation if offer.offer_id == offer_id][0].product_id
        relevant_competitor_offers = [offer for offer in market_situation if
                                      offer.product_id == product_id and
                                      offer.merchant_id != merchant.merchant_id]
        if not relevant_competitor_offers:
            return merchant.settings['upper_price_bound']

        cheapest_offer = min(relevant_competitor_offers, key=lambda offer: offer.price)
        if cheapest_offer.price <= merchant.settings['lower_price_bound'] or \
                cheapest_offer.price > merchant.settings['upper_price_bound']:
            return merchant.settings['upper_price_bound']
        else:
            return cheapest_offer.price - merchant.settings['price_decrement']


class Merchant(PricewarsMerchant):
    def __init__(self, token: Optional[str], port: int, marketplace_url: str, producer_url: str, strategy, merchant_name: str):
        super().__init__(port, token, marketplace_url, producer_url, merchant_name)
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
                        help="Chose the merchant's strategy (example: Cheapest, \"Two Bound\")")
    parser.add_argument('--name', type=str, default=None,
                        help="The merchant's name. Defaults to the name of the strategy.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    strategies = {
        CheapestStrategy.name: CheapestStrategy,
        TwoBoundStrategy.name: TwoBoundStrategy,
    }

    merchant_name = args.name
    if merchant_name is None:
        merchant_name = args.strategy

    merchant = Merchant(args.token, args.port, args.marketplace, args.producer, strategies[args.strategy], merchant_name)
    merchant.run()


if __name__ == '__main__':
    main()

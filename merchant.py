import argparse
from typing import Optional

from api import Marketplace, Producer
from pricewars_merchant import PricewarsMerchant


class Merchant(PricewarsMerchant):
    def __init__(self, token: Optional[str], port: int, marketplace_url: str, producer_url: str, name='Cheapest'):
        super().__init__(port, token, marketplace_url, producer_url, name)

        self.settings.update({
            'price decrement': 0.05,
            'default price': 30
        })

    def calculate_price(self, offer_id, market_situation):
        product_id = [offer for offer in market_situation if offer.offer_id == offer_id][0].product_id
        relevant_competitor_offers = [offer for offer in market_situation if
                                      offer.product_id == product_id and
                                      offer.merchant_id != self.merchant_id]
        if not relevant_competitor_offers:
            return self.settings['default price']

        cheapest_offer = min(relevant_competitor_offers, key=lambda offer: offer.price)
        return cheapest_offer.price - self.settings['price decrement']


def parse_arguments():
    parser = argparse.ArgumentParser(description='PriceWars Merchant')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--port', type=int, help='port to bind flask App to')
    group.add_argument('--token', type=str, help='Merchant secret token')
    parser.add_argument('--marketplace', type=str, default=Marketplace.DEFAULT_URL, help='Marketplace URL')
    parser.add_argument('--producer', type=str, default=Producer.DEFAULT_URL, help='Producer URL')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    merchant = Merchant(args.token, args.port, args.marketplace, args.producer)
    merchant.run()

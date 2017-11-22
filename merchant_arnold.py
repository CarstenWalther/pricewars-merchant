import argparse
import threading
import time

from pricewars import MerchantServer
from pricewars.api import Marketplace, Producer
from pricewars.models import SoldOffer


class ArnoldMerchant:
    def __init__(self, port: int, marketplace_url: str, producer_url: str):
        self.server_thread = self.setup_server(port)

        self.marketplace = Marketplace(marketplace_url)
        self.marketplace.wait_for_host()
        response = self.marketplace.register(endpoint_url_or_port=port, merchant_name='Arnold', algorithm_name='ss policy')
        self.merchant_id = response.merchant_id
        self.token = response.merchant_token
        self.producer = Producer(self.token, producer_url)

    def setup_server(self, port):
        server = MerchantServer(self)
        thread = threading.Thread(target=server.app.run, kwargs={'host': '0.0.0.0', 'port': port})
        thread.daemon = True
        thread.start()
        return thread

    def update(self):
        print('update')

    def run(self):
        while True:
            self.update()
            time.sleep(1)

    def sold_offer(self, offer: SoldOffer):
        pass

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

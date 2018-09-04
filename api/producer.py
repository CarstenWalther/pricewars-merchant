from typing import List

from api.pricewars_base_api import PricewarsBaseApi
from models import ProductInfo
from models import Order


class Producer(PricewarsBaseApi):
    DEFAULT_URL = 'http://localhost:3050'

    def __init__(self, token: str, host: str = DEFAULT_URL, debug: bool = False):
        super().__init__(token, host, debug)

    def order(self, quantity) -> Order:
        r = self.request('post', 'orders', data={'quantity': quantity})
        return Order.from_dict(r.json())

    def get_products_info(self) -> List[ProductInfo]:
        r = self.request('get', 'products')
        return ProductInfo.from_list(r.json())

    def get_product_info(self, product_id) -> ProductInfo:
        r = self.request('get', 'products/{}'.format(product_id))
        return ProductInfo.from_dict(r.json())

from typing import List

from api.pricewars_base_api import PricewarsBaseApi
from models import Product
from models import ProductInfo
from models import Order


class Producer(PricewarsBaseApi):
    DEFAULT_URL = 'http://producer:3050'

    def __init__(self, token: str, host: str = DEFAULT_URL, debug: bool = False):
        super().__init__(token, host, debug)

    def order(self, amount) -> Order:
        r = self.request('post', 'orders', data={'amount': amount})
        return Order.from_dict(r.json())

    def get_products(self) -> List[ProductInfo]:
        r = self.request('get', 'products')
        return ProductInfo.from_list(r.json())

    def add_products(self, products: List[Product]):
        product_dict_list = [p.to_dict() for p in products]
        self.request('post', 'products', json=product_dict_list)

    def update_products(self, products: List[Product]):
        product_dict_list = [p.to_dict() for p in products]
        self.request('put', 'products', json=product_dict_list)

    def get_product(self, product_uid) -> ProductInfo:
        r = self.request('get', 'products/{}'.format(product_uid))
        return ProductInfo.from_dict(r.json())

    def add_product(self, product: Product):
        self.request('post', 'products', json=product.to_dict())

    def update_product(self, product: Product):
        self.request('put', 'products/{}'.format(product.uid), json=product.to_dict())

    def delete_product(self, product_uid):
        self.request('delete', 'products/{}'.format(product_uid))

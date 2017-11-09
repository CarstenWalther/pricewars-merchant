from typing import List

from .PricewarsBaseApi import PricewarsBaseApi
from merchant_sdk.models import Product
from merchant_sdk.models import Order


class ProducerApi(PricewarsBaseApi):
    DEFAULT_URL = 'http://producer:3050'

    def __init__(self, token: str, host: str=DEFAULT_URL, debug: bool=False):
        super().__init__(token, host, debug)

    def buy_product(self) -> Order:
        return self.buy_products(amount=1)

    def buy_products(self, amount) -> Order:
        r = self.request('post', 'buy/{}'.format(amount))
        return Order.from_dict(r.json())

    def get_products(self) -> List[Product]:
        r = self.request('get', 'products')
        return Product.from_list(r.json())

    def add_products(self, products: List[Product]):
        product_dict_list = [p.to_dict() for p in products]
        self.request('post', 'products', json=product_dict_list)

    def update_products(self, products: List[Product]):
        product_dict_list = [p.to_dict() for p in products]
        self.request('put', 'products', json=product_dict_list)

    def get_product(self, product_uid) -> Product:
        r = self.request('get', 'products/{}'.format(product_uid))
        return Product.from_dict(r.json())

    def add_product(self, product: Product):
        self.request('post', 'products', json=product.to_dict())

    def update_product(self, product: Product):
        self.request('put', 'products/{}'.format(product.uid), json=product.to_dict())

    def delete_product(self, product_uid):
        self.request('delete', 'products/{}'.format(product_uid))

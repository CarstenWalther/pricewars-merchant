from typing import Union
from typing import Optional

from models.PricewarsObject import PricewarsObject
from models import Product


class Order(PricewarsObject):
    def __init__(self, billing_amount: float, stock: int, product: Union[Product, dict],
                 left_in_stock: Optional[int] = None):
        self.billing_amount = billing_amount
        self.stock = stock
        self.left_in_stock = left_in_stock
        if type(product) == dict:
            self.product = Product.from_dict(product)
        else:
            self.product = product

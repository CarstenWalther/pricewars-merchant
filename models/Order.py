from typing import Union
from typing import Optional

from models.PricewarsObject import PricewarsObject
from models import Product


class Order(PricewarsObject):
    def __init__(self, billing_amount: float, fixed_cost: float, unit_price: float, stock: int,
                 product: Union[Product, dict], left_in_stock: Optional[int] = None):
        """
        :param billing_amount: Is the same as product.amount * unit_price + fixed_cost.
        """
        self.billing_amount = billing_amount
        self.fixed_cost = fixed_cost
        self.unit_price = unit_price
        self.stock = stock
        self.left_in_stock = left_in_stock
        if type(product) == dict:
            self.product = Product.from_dict(product)
        else:
            self.product = product

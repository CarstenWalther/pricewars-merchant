from typing import List
from typing import Optional

from models.PricewarsObject import PricewarsObject
from models import Product


class Order(PricewarsObject):
    def __init__(self, product_id: int, product_name: str, billing_amount: float, fixed_cost: float, unit_price: float,
                 stock: int, products: List[dict], left_in_stock: Optional[int] = None):
        """
        :param billing_amount: Is the same as number of ordered products * unit_price + fixed_cost.
        """
        self.product_id = product_id
        self.product_name = product_name
        self.billing_amount = billing_amount
        self.fixed_cost = fixed_cost
        self.unit_price = unit_price
        self.stock = stock
        self.left_in_stock = left_in_stock
        self.products: List[Product] = Product.from_list(products)

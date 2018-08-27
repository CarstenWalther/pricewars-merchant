from typing import Collection

from .PricewarsObject import PricewarsObject


class ProductInfo(PricewarsObject):
    def __init__(self, product_id: int, name: str, unit_price: float, fixed_order_cost: float, stock: int,
                 qualities: Collection[int], time_to_live=-1, start_of_lifetime=-1):
        self.product_id = product_id
        self.name = name
        self.unit_price = unit_price
        self.fixed_order_cost = fixed_order_cost
        self.stock = stock
        self.qualities = set(qualities)
        self.time_to_live = time_to_live
        self.start_of_lifetime = start_of_lifetime

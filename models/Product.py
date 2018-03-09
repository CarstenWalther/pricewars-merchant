from .PricewarsObject import PricewarsObject


class Product(PricewarsObject):
    def __init__(self, uid=-1, product_id=-1, name='', quality=0, amount=1, signature='', time_to_live=-1,
                 start_of_lifetime=-1):
        self.uid = uid
        self.product_id = product_id
        self.name = name
        self.quality = quality
        self.amount = amount
        self.signature = signature
        self.time_to_live = time_to_live
        self.start_of_lifetime = start_of_lifetime


class ProductInfo(PricewarsObject):
    def __init__(self, uid=-1, product_id=-1, name='', quality=0, price=0, fixed_order_cost=0, stock=-1,
                 time_to_live=-1, start_of_lifetime=-1):
        self.uid = uid
        self.product_id = product_id
        self.name = name
        self.quality = quality
        self.price = price
        self.fixed_order_cost = fixed_order_cost
        self.stock = stock
        self.time_to_live = time_to_live
        self.start_of_lifetime = start_of_lifetime

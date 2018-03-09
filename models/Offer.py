from typing import Optional

from models.PricewarsObject import PricewarsObject
from models import Product


class Offer(PricewarsObject):
    def __init__(self, offer_id=-1, uid=-1, product_id=-1, quality=0, merchant_id: Optional[str] = None, amount=1,
                 price=0.0, shipping_time=None, prime=False, signature=''):
        self.offer_id = offer_id
        self.uid = uid
        self.product_id = product_id
        self.quality = quality
        self.merchant_id = merchant_id
        self.amount = amount
        self.price = price
        self.shipping_time = shipping_time or {'standard': 3}
        self.prime = prime
        self.signature = signature

    @staticmethod
    def from_product(product: Product, price: float, shipping_time: dict, prime: bool = False) -> 'Offer':
        return Offer(
            uid=product.uid,
            product_id=product.product_id,
            quality=product.quality,
            amount=product.amount,
            price=price,
            shipping_time=shipping_time,
            prime=prime,
            signature=product.signature,
        )

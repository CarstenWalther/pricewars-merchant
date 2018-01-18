from time import sleep

from pricewars.api import Marketplace, Producer, Kafka
from pricewars.models import Offer

marketplace = Marketplace()
marketplace.wait_for_host()
r = marketplace.register(endpoint_url_or_port=5009, merchant_name='TEST', algorithm_name='test')
merchant_id = r.merchant_id
token = r.merchant_token
print('token', token)

producer = Producer(token)
print(producer.get_products())
order = producer.order(1000)
shipping_time = {
    'standard': 5,
    'prime': 1
}
offer = Offer.from_product(order.product, 20, shipping_time)
marketplace.add_offer(offer)

#sleep(10)

kafka_reverse_proxy = Kafka(token)
csv = kafka_reverse_proxy.download_topic_data('buyOffer')
print(csv)

csv = kafka_reverse_proxy.download_topic_data('marketSituation')
print(csv)
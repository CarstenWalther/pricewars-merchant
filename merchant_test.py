from time import sleep
import pandas as pd
import numpy as np

from pricewars.api import Marketplace, Producer, Kafka
from pricewars.models import Offer
"""
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
"""
token = 'BIN3xmcArXIcDUN994Mqa24f9lfBH5vSNJVM78WyfobsJy2R3NJdPNJY9yjQ5ftS'
kafka_reverse_proxy = Kafka(token)
csv = kafka_reverse_proxy.download_topic_data('buyOffer')
csv = csv[['timestamp', 'amount', 'price']]
print(csv)
#print(csv.dtypes)
#csv = kafka_reverse_proxy.download_topic_data('marketSituation')
#print(csv)
#print(csv.dtypes)
interval = 1
csv = csv.set_index('timestamp').resample(str(interval) + 's').agg({'amount' : 'sum', 'price': 'min'})\
    .rename(columns={'amount': 'sales', 'price': 'min_price'}).fillna({'sales': 0, 'min_price': 30})
print(csv)

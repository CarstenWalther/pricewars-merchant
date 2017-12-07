from pricewars.api import Marketplace, Producer

marketplace = Marketplace()
marketplace.wait_for_host()
r = marketplace.register(endpoint_url_or_port=5009, merchant_name='TEST', algorithm_name='test')
merchant_id = r.merchant_id
token = r.merchant_token
print('token', token)

producer = Producer(token)
print(producer.get_products())
order = producer.order(1000)
print(order)

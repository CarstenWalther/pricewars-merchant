from typing import List
from typing import Optional
from typing import Union
import socket
from urllib.parse import urlparse

from api.pricewars_base_api import PricewarsBaseApi
from models import Offer, MerchantRegisterResponse


class Marketplace(PricewarsBaseApi):
    DEFAULT_URL = 'http://marketplace:8080'

    def __init__(self, token: Optional[str] = None, host: str = DEFAULT_URL, debug: bool = False):
        super().__init__(token, host, debug)

    def get_offers(self, include_empty_offers: bool = False) -> List[Offer]:
        r = self.request('get', 'offers', params={'include_empty_offer': include_empty_offers})
        return Offer.from_list(r.json())

    def add_offer(self, offer: Offer) -> Offer:
        r = self.request('post', 'offers', json=offer.to_dict())
        return Offer.from_dict(r.json())

    def update_offer(self, offer: Offer):
        self.request('put', 'offers/{:d}'.format(offer.offer_id), json=offer.to_dict())

    def restock(self, offer_id: int, amount: int = 0, signature: str = '') -> None:
        body = {
            'amount': amount,
            'signature': signature
        }
        self.request('patch', 'offers/{:d}/restock'.format(offer_id), json=body)

    def holding_cost_rate(self) -> float:
        """
        :return: The holding cost per unit in the inventory per minute
        """
        response = self.request('get', 'holding_cost_rate')
        return response.json()

    def register(self, endpoint_url_or_port: Union[str, int], merchant_name: str,
                 algorithm_name: str = '') -> MerchantRegisterResponse:
        if type(endpoint_url_or_port) == int:
            port = endpoint_url_or_port
            endpoint_url = 'http://{}:{}'.format(self._get_own_ip_address(self.host), port)
        else:
            endpoint_url = endpoint_url_or_port

        body = {
            'api_endpoint_url': endpoint_url,
            'merchant_name': merchant_name,
            'algorithm_name': algorithm_name
        }
        r = self.request('post', 'merchants', json=body)
        response = MerchantRegisterResponse.from_dict(r.json())
        self.set_auth_token(response.merchant_token)
        return response

    def unregister(self, merchant_token: str = '') -> None:
        self.request('delete', 'merchants/{:s}'.format(merchant_token))

    @staticmethod
    def _get_own_ip_address(destination: str):
        try:
            url = urlparse(destination)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((url.hostname, url.port or 80))
            return s.getsockname()[0]
        except socket.gaierror:
            raise RuntimeError("Cannot connect to " + destination)

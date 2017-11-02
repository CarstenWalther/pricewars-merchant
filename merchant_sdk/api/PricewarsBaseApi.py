from urllib.parse import urljoin
import requests
from time import time
from typing import Optional

from merchant_sdk.models.ApiError import ApiError


class PricewarsBaseApi:
    def __init__(self, token: Optional[str], host: str, debug: bool):
        self.host = host if '://' in host else 'http://' + host
        self.debug = debug
        self.session = requests.Session()
        if token is not None:
            self.set_auth_token(token)

    def request(self, method: str, resource: str, **kwargs):
        if self.debug:
            print('request', self.__class__, method, resource, kwargs)

        url = urljoin(self.host, resource)
        response = self.session.request(method, url, **kwargs)

        if self.debug:
            print('response', 'status({:d})'.format(response.status_code), response.text)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ApiError(response.status_code, url, response.text) from None
        return response

    def set_auth_token(self, token: str):
        self.session.headers.update({'Authorization': 'Token {:s}'.format(token)})

    def wait_for_host(self, timeout: int = 60) -> None:
        """
        Waits until it is possible to connect to host.
        """
        start = time()
        while time() - start < timeout:
            try:
                self.session.get(self.host)
                return
            except requests.exceptions.ConnectionError:
                pass
        raise RuntimeError(self.host + ' not reachable')
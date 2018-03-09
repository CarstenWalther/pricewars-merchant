from time import time
from typing import Optional
from urllib.parse import urljoin
import requests
import logging

from api.ApiError import ApiError


class PricewarsBaseApi:
    def __init__(self, token: Optional[str], host: str, debug: bool):
        self.host = host if '://' in host else 'http://' + host
        self.session = requests.Session()
        if token is not None:
            self.set_auth_token(token)

        logging.basicConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.WARNING)

    def request(self, method: str, resource: str, **kwargs):
        self.logger.debug(' '.join((method, resource, str(kwargs))))
        url = urljoin(self.host, resource)
        response = self.session.request(method, url, **kwargs)
        self.logger.debug(str(response.status_code) + ' ' + response.text)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            raise ApiError(response.status_code, url, response.text) from None
        return response

    def set_auth_token(self, token: str):
        self.session.headers.update({'Authorization': 'Token {:s}'.format(token)})

    def wait_for_host(self, timeout: float = 60) -> None:
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
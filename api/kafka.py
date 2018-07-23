from typing import Optional
import pandas as pd

from api.pricewars_base_api import PricewarsBaseApi


class Kafka(PricewarsBaseApi):
    DEFAULT_URL = 'http://localhost:8001'

    def __init__(self, token: str, host: str = DEFAULT_URL, debug: bool = False):
        super().__init__(token, host, debug)

    def _request_topic_url(self, topic: str) -> str:
        response = self.request('get', 'export/data/{:s}'.format(topic))
        return '{:s}/{:s}'.format(self.host, response.json()['url'])

    def download_topic_data(self, topic: str) -> Optional[pd.DataFrame]:
        url = self._request_topic_url(topic)
        try:
            return pd.read_csv(url, parse_dates=['timestamp'])
        except pd.errors.EmptyDataError:
            return None

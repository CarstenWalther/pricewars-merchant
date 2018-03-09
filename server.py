import json
import logging

from flask import Flask, request, Response
from flask_cors import CORS

import pricewars_merchant
from models import SoldOffer


def json_response(message):
    return Response(json.dumps(message), status=200, mimetype='application/json')


class MerchantServer:

    def __init__(self, merchant: 'pricewars_merchant.PricewarsMerchant', logging_level=logging.WARNING):
        self.merchant = merchant
        self.app = Flask(__name__)
        CORS(self.app)

        logging.basicConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        self.register_routes()

    def register_routes(self):
        self.app.add_url_rule('/settings', 'get_settings', self.get_settings, methods=['GET'])
        self.app.add_url_rule('/settings', 'put_settings', self.update_settings, methods=['PUT', 'POST'])
        self.app.add_url_rule('/settings/execution', 'set_state', self.set_state, methods=['POST'])
        self.app.add_url_rule('/settings/execution', 'get_state', self.get_state, methods=['GET'])
        self.app.add_url_rule('/sold', 'item_sold', self.item_sold, methods=['POST'])

    def get_settings(self):
        return json_response(self.merchant.settings)

    def update_settings(self):
        self.merchant.update_settings(request.json)
        self.logger.debug('Update settings ' + str(self.merchant.settings))
        return self.get_settings()

    def get_state(self):
        return json_response({'state': self.merchant.state})

    def set_state(self):
        next_state = request.json['nextState']
        self.logger.debug('Execution setting - next state: ' + next_state)

        if next_state == 'start':
            self.merchant.start()
        elif next_state == 'stop':
            self.merchant.stop()

        return json_response({})

    def item_sold(self):
        try:
            offer = SoldOffer.from_dict(request.get_json(force=True))
            self.merchant.sold_offer(offer)
        except Exception as e:
            self.logger.error(e)

        return json_response({})

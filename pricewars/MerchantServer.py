import json
import logging

from flask import Flask, request, Response
from flask_cors import CORS

from .PricewarsMerchant import PricewarsMerchant
from .models import SoldOffer


def json_response(obj):
    js = json.dumps(obj)
    resp = Response(js, status=200, mimetype='application/json')
    return resp


class MerchantServer:
    
    def __init__(self, merchant: PricewarsMerchant, debug=False):
        self.merchant = merchant
        self.app = Flask(__name__)
        CORS(self.app)

        logging.basicConfig()
        self.logger = logging.getLogger('MerchantServer')
        self.logger.setLevel(logging.DEBUG if debug else logging.WARNING)

        self.register_routes()

    '''
        Helper methods
    '''

    def get_all_settings(self):
        settings = dict(self.merchant.get_settings())
        settings['state'] = self.merchant.get_state()
        return settings

    def update_all_settings(self, new_settings):
        new_logic_settings = {k: new_settings[k] for k in new_settings if k in self.merchant.get_settings()}
        self.merchant.update_settings(new_logic_settings)
        self.logger.debug('update settings ' + str(self.get_all_settings()))

    '''
        Routes
    '''

    def register_routes(self):
        self.app.add_url_rule('/settings', 'get_settings', self.get_settings, methods=['GET'])
        self.app.add_url_rule('/settings', 'put_settings', self.put_settings, methods=['PUT', 'POST'])
        self.app.add_url_rule('/settings/execution', 'set_state', self.set_state, methods=['POST'])
        self.app.add_url_rule('/sold', 'item_sold', self.item_sold, methods=['POST'])

    '''
        Endpoint definitions
    '''

    def get_settings(self):
        return json_response(self.get_all_settings())

    def put_settings(self):
        new_settings = request.json
        self.update_all_settings(new_settings)
        return json_response(self.get_all_settings())

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
            sent_json = request.get_json(force=True)
            offer = SoldOffer.from_dict(sent_json)
            self.merchant.sold_offer(offer)
        except Exception as e:
            self.logger.error(e)

        return json_response({})

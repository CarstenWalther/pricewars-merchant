from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy.stats import poisson


def learn_demand_function(X_train, y_train):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    def demand_distribution(demand, features):
        mean = model.predict(features)
        mean[mean < 0] = 0
        return poisson.pmf(demand.reshape((1, -1)), mean.reshape((-1, 1)))

    return demand_distribution


def aggregate_sales_to_market_situations(sales_data, market_situations):
    """
    This function sums up all sales that happen between each two successive market situations.
    The results are divided by the time between the two market situations to make
    them independent from the interval length.
    Sales are counted for each offer separately.
    """
    grouped = sales_data.groupby(
        ['offer_id', pd.cut(sales_data['timestamp'], market_situations['timestamp'].unique(), right=False)])
    sales_by_interval = grouped['amount'].sum()
    # Calculate the time span from the start and end of the interval
    time_spans = sales_by_interval.index.get_level_values('timestamp') \
        .map(lambda e: e.right - e.left).astype('timedelta64[ns]').values
    sales_per_minute = sales_by_interval / (time_spans / np.timedelta64(1, 'm'))
    return sales_per_minute


def extract_features(market_situation, own_offer_id):
    own_offer = market_situation.loc[own_offer_id]
    competitor_offers = market_situation.loc[market_situation.index != own_offer_id]
    price_rank = (competitor_offers['price'] <= own_offer['price']).sum() if competitor_offers.size != 0 else 0
    return own_offer['price'], price_rank


def aggregate_sales_data(merchant_id, market_situations, sales_data):
    """
    This function creates a pair of features and sales for each market situation and offer.
    It returns a dictionary with product ids as keys and lists of the mentioned pairs as values.
    """
    sales_per_minute = aggregate_sales_to_market_situations(sales_data, market_situations)
    # We want to look up values with the timestamp of a market situation.
    # Thus the interval index is transformed to a timestamp index.
    sales_per_minute.index = sales_per_minute.index.map(lambda e: (e[0], e[1].left))
    sales_data_by_product = defaultdict(list)

    # We look at each market situation (same timestamp) and separate market data for each product.
    for (product_id, timestamp), market_situation in market_situations.groupby(['product_id', 'timestamp']):
        market_situation = market_situation.set_index(['offer_id'])
        # A market situation can have multiple offers that belong to this merchant.
        # For each own offer a feature-sales-pair is generated that
        # assumes that all other offers are competitors offers.
        for own_offer_id, _ in market_situation[market_situation["merchant_id"] == merchant_id].iterrows():
            features = extract_features(market_situation, own_offer_id)
            sales = sales_per_minute.get((own_offer_id, timestamp), default=0)
            sales_data_by_product[product_id].append((features, sales))

    # We cannot tell the sales per minute for the last market situation.
    # That is why the last trainings pair is removed.
    # TODO: What if last trainings pair does not belong to last market situation? There must be a better way
    for product_id in sales_data_by_product:
        sales_data_by_product[product_id] = sales_data_by_product[product_id][:-1]

    return sales_data_by_product


def demand_learning(market_situations, sales_data, merchant_id, decision_interval_in_seconds):
    if market_situations is not None and sales_data is not None:
        sales_per_product = aggregate_sales_data(merchant_id, market_situations, sales_data)
        # Currently there is only one product type
        if sales_per_product[1]:
            features, sales_per_minute = zip(*sales_per_product[1])
            sales_per_decision_interval = np.array(sales_per_minute) / 60 * decision_interval_in_seconds
            return learn_demand_function(features, sales_per_decision_interval)
    return None

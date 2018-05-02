import numpy as np
import pandas as pd

from policy.demand_learning import extract_features


def default_order_policy(stock):
    return 20 if stock == 0 else 0


def default_pricing_policy(selling_price_low, selling_price_high):
    return lambda stock: np.random.randint(selling_price_low, selling_price_high + 1)


class PolicyOptimizer:
    def __init__(self, max_stock=40, selling_price_low=20, selling_price_high=40):
        self.max_stock = max_stock
        self.selling_price_low = selling_price_low
        self.selling_price_high = selling_price_high
        self.remaining_stock = np.arange(self.max_stock + 1)
        self.order_quantity = np.arange(self.max_stock + 1)
        self.selling_prices = np.arange(self.selling_price_low, self.selling_price_high + 1)
        self.demand = np.arange(self.max_stock + 1)
        self.expected_profits = np.zeros(max_stock + 1)

    def create_policies(self, demand_distribution, product_cost, fixed_order_cost, holding_cost_per_interval,
                        market_situation, own_offer_id, max_iterations=5):
        if not demand_distribution:
            print('Use default policy')
            return default_order_policy, default_pricing_policy(self.selling_price_low, self.selling_price_high)

        order_policy = np.zeros(len(self.remaining_stock))
        pricing_policy = np.zeros(len(self.remaining_stock))

        market_situation = pd.DataFrame([offer.to_dict() for offer in market_situation])
        market_situation.set_index(['offer_id'], inplace=True)

        for _ in range(max_iterations):
            old_order_policy = order_policy
            old_price_policy = pricing_policy
            order_policy, pricing_policy, self.expected_profits = \
                policy_optimization(demand_distribution, product_cost, fixed_order_cost, holding_cost_per_interval,
                                    self.selling_prices, self.expected_profits, self.remaining_stock,
                                    self.order_quantity, self.demand, market_situation, own_offer_id, iterations=40)

            #print(order_policy)
            #print(pricing_policy)

            # Ignore the first price (i.e. price at inventory level 0) because no items can be offered.
            # The first price will fluctuate a lot and delay the policy convergence
            if np.array_equal(order_policy, old_order_policy) and np.array_equal(pricing_policy[1:], old_price_policy[1:]):
                # The policy has converged
                break

            self.order_quantity = adapt_order_search_space(order_policy)
            self.selling_prices = adapt_price_search_space(pricing_policy)

        def order_policy_function(stock):
            return order_policy[np.clip(stock, 0, len(order_policy) - 1)]

        def pricing_policy_function(stock):
            return pricing_policy[np.clip(stock, 0, len(pricing_policy) - 1)]

        if not pricing_policy.any():
            print('Warning: avoid selling products for 0€. Use default pricing policy')
            return order_policy_function, default_pricing_policy(self.selling_price_low, self.selling_price_high)

        return order_policy_function, pricing_policy_function


def get_features(selling_prices, market_situation, own_offer_id):
    features = []
    # TODO: don't save features as list; preallocate numpy array
    for price in selling_prices:
        market_situation.at[own_offer_id, 'price'] = price
        features.append(extract_features(market_situation, own_offer_id))
    return np.array(features)


def policy_optimization(demand_distribution, product_cost, fixed_order_cost, holding_cost_per_interval, selling_prices,
                        expected_profits, remaining_stock, order_quantity, demand, market_situation, own_offer_id,
                        iterations):
    remaining_stock_reshaped = remaining_stock.reshape((-1, 1, 1, 1))
    order_quantity_reshaped = order_quantity.reshape((1, -1, 1, 1))
    selling_prices_reshaped = selling_prices.reshape((1, 1, -1, 1))
    demand_reshaped = demand.reshape((1, 1, 1, -1))

    features = get_features(selling_prices, market_situation.copy(), own_offer_id)
    probabilities = demand_distribution(demand, features).reshape(1, 1, len(features), -1)
    sales = np.minimum(demand_reshaped, remaining_stock_reshaped)

    for i in range(1, iterations + 1):
        all_expected_profits = np.sum(probabilities * (
                profit(remaining_stock_reshaped, sales, order_quantity_reshaped, selling_prices_reshaped, product_cost,
                       fixed_order_cost, holding_cost_per_interval)
                + i * expected_profits[
                    np.minimum(remaining_stock_reshaped + order_quantity_reshaped - sales, len(expected_profits) - 1)]
        ), axis=3) / (i + 1)
        expected_profits = np.amax(all_expected_profits, axis=(1, 2))

    # Combine order_quantity and price dimension, because we cannot get argmax over multiple dimensions
    policy = np.argmax(all_expected_profits.reshape(len(remaining_stock), -1), axis=1)
    order_policy_indices, price_policy_indices = np.unravel_index(policy, (len(order_quantity), len(selling_prices)))
    order_policy = order_quantity[order_policy_indices]
    price_policy = selling_prices[price_policy_indices]
    return order_policy, price_policy, expected_profits


def profit(remaining_stock, sales, order_quantity, selling_price, product_cost, fixed_order_cost,
           holding_cost_per_interval):
    return sales_revenue(sales, selling_price) - order_cost(order_quantity, product_cost, fixed_order_cost) \
           - holding_cost(remaining_stock, holding_cost_per_interval)


def order_cost(order_quantity, product_cost, fixed_order_cost):
    return order_quantity * product_cost + (order_quantity > 0) * fixed_order_cost


def holding_cost(remaining_stock, holding_cost_per_interval):
    return remaining_stock * holding_cost_per_interval


def sales_revenue(sales, selling_price):
    return sales * selling_price


def adapt_order_search_space(order_policy):
    order_buffer = 5
    # When calculating the minimum order size, ignore non-orders, i.e. orders of zero products
    orders_greater_zero = order_policy[order_policy > 0]
    lowest_order = np.amin(orders_greater_zero) if orders_greater_zero.size > 0 else 1
    min_order = max(1, lowest_order - order_buffer)
    max_order = np.amax(order_policy) + order_buffer

    # The array must always contain a zero so that the merchant is able to not order products.
    # Make some space in the array and set the first element to zero.
    new_order_quantity = np.arange(min_order - 1, max_order + 1)
    new_order_quantity[0] = 0
    return new_order_quantity


def adapt_price_search_space(pricing_policy):
    price_buffer = 5
    selling_price_low = max(0, np.amin(pricing_policy) - price_buffer)
    selling_price_high = np.amax(pricing_policy) + price_buffer
    return np.arange(selling_price_low, selling_price_high + 1)

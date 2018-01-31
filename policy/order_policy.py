import numpy as np


def create_policy(demand_distribution, product_cost, fixed_order_cost, holding_cost_per_interval,
                  max_stock, selling_price_low, selling_price_high, threshold=1e-5):
    """
    Uses a known (or predicted) demand distribution and the Bellman equation to determine an order policy
    """
    expected_profits = np.zeros(max_stock + 1)
    remaining_stock, order_quantity, price, demand = np.split(
        np.mgrid[0:max_stock + 1, 0:max_stock + 1, selling_price_low:selling_price_high + 1, 0:max_stock + 1], 4)
    remaining_stock = np.squeeze(remaining_stock)
    order_quantity = np.squeeze(order_quantity)
    demand = np.squeeze(demand)
    price = np.squeeze(price)
    probabilities = demand_distribution(demand, price)
    sales = np.minimum(demand, remaining_stock + order_quantity)
    all_expected_profits = None
    difference = float('inf')
    iteration = 0

    # Stop iterating if changes in expected profit are below the threshold
    while difference > threshold:
        all_expected_profits = np.sum(probabilities * (
                profit(remaining_stock, sales, order_quantity, price, product_cost, fixed_order_cost,
                       holding_cost_per_interval)
                + iteration * expected_profits[np.clip(remaining_stock + order_quantity - sales, 0, max_stock)]
        ), axis=3) / (iteration + 1)
        old_expected_profits = expected_profits
        expected_profits = np.max(all_expected_profits, axis=(1, 2))
        difference = np.max(np.absolute(expected_profits - old_expected_profits))
        iteration += 1

    new_shaped_e_profits = all_expected_profits.reshape(max_stock + 1, -1)
    policy = np.argmax(new_shaped_e_profits, axis=-1)
    order_policy, price_policy = np.unravel_index(policy, (max_stock + 1, selling_price_high - selling_price_low + 1))
    # Transform from index to actual value
    price_policy += selling_price_low
    return order_policy, price_policy


def profit(remaining_stock, sales, order_quantity, selling_price, product_cost, fixed_order_cost,
           holding_cost_per_interval):
    return sales_revenue(sales, selling_price) - order_cost(order_quantity, product_cost, fixed_order_cost) \
           - holding_cost(remaining_stock, order_quantity, holding_cost_per_interval)


def order_cost(order_quantity, product_cost, fixed_order_cost):
    return order_quantity * product_cost + (order_quantity > 0) * fixed_order_cost


def holding_cost(remaining_stock, order_quantity, holding_cost_per_interval):
    return (remaining_stock + order_quantity) * holding_cost_per_interval


def sales_revenue(sales, selling_price):
    return sales * selling_price


def generic_metric(metric, remaining_stock, sales, order_quantity, selling_price, product_cost, fixed_order_cost,
                   holding_cost_per_interval):
    if metric == 'profit':
        return profit(remaining_stock, sales, order_quantity, selling_price, product_cost, fixed_order_cost,
                      holding_cost_per_interval)
    elif metric == 'revenue':
        return sales_revenue(sales, selling_price)
    elif metric == 'holding_cost':
        holding_cost(remaining_stock, order_quantity, holding_cost_per_interval)
    elif metric == 'order_cost':
        order_cost(order_quantity, product_cost, fixed_order_cost)
    else:
        raise ValueError('Invalid metric')


def evaluate_policy(policy, metric, demand_distribution, selling_price, product_cost, fixed_order_cost,
                    holding_cost_per_interval, max_stock, threshold=1e-5):
    """
    For metric use 'profit', 'revenue', 'holding_cost' or 'order_cost'
    """
    expected_profits = np.zeros(max_stock + 1)
    remaining_stock, demand = np.split(np.mgrid[0:max_stock + 1, 0:max_stock + 1], 2)
    remaining_stock = np.squeeze(remaining_stock)
    order_quantity = policy[remaining_stock]
    demand = np.squeeze(demand)
    probabilities = demand_distribution(demand)
    sales = np.minimum(demand, remaining_stock + order_quantity)
    difference = float('inf')
    iteration = 0

    # Stop iterating if changes in expected profit are below the threshold
    while difference > threshold:
        old_expected_profits = expected_profits
        expected_profits = np.sum(probabilities * (
                generic_metric(metric, remaining_stock, sales, order_quantity, selling_price, product_cost,
                               fixed_order_cost, holding_cost_per_interval)
                + iteration * expected_profits[np.clip(remaining_stock + order_quantity - sales, 0, max_stock)]
        ), axis=-1) / (iteration + 1)
        difference = np.max(np.absolute(expected_profits - old_expected_profits))
        iteration += 1

    return expected_profits[0]

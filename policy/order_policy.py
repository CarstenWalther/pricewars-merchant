import numpy as np


def create_policy(demand_distribution, selling_price, product_cost, fixed_order_cost, holding_cost_per_interval,
                  max_stock, threshold=1e-5):
    """
    Uses a known (or predicted) demand distribution and the Bellman equation to determine an order policy
    """
    expected_profits = np.zeros(max_stock + 1)
    remaining_stock, order_quantity, demand = np.split(
        np.mgrid[0:max_stock + 1, 0:max_stock + 1, 0:max_stock + 1], 3)
    remaining_stock = np.squeeze(remaining_stock)
    order_quantity = np.squeeze(order_quantity)
    demand = np.squeeze(demand)
    probabilities = demand_distribution(demand)
    sales = np.minimum(demand, remaining_stock + order_quantity)
    all_expected_profits = None
    difference = float('inf')
    iteration = 0

    # Stop iterating if changes in expected profit are below the threshold
    while difference > threshold:
        all_expected_profits = np.sum(probabilities * (
                profit(remaining_stock, sales, order_quantity, selling_price, product_cost, fixed_order_cost,
                       holding_cost_per_interval)
                + iteration * expected_profits[np.clip(remaining_stock + order_quantity - sales, 0, max_stock)]
        ), axis=-1) / (iteration + 1)
        old_expected_profits = expected_profits
        expected_profits = np.max(all_expected_profits, axis=-1)
        difference = np.max(np.absolute(expected_profits - old_expected_profits))
        iteration += 1

    policy = np.argmax(all_expected_profits, axis=-1)
    print('policy:', policy)
    print('expected profit', expected_profits[0])

    def policy_function(remaining_stock):
        return policy[np.clip(remaining_stock, 0, max_stock)]

    # TODO: what to return?
    return policy_function, policy


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


def evaluate_policy(policy, demand_distribution, selling_price, product_cost, fixed_order_cost,
                    holding_cost_per_interval, max_stock, threshold=1e-5):
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
                profit(remaining_stock, sales, order_quantity, selling_price, product_cost, fixed_order_cost,
                       holding_cost_per_interval)
                #sales_revenue(sales, selling_price)
                + iteration * expected_profits[np.clip(remaining_stock + order_quantity - sales, 0, max_stock)]
        ), axis=-1) / (iteration + 1)
        difference = np.max(np.absolute(expected_profits - old_expected_profits))
        iteration += 1

    return expected_profits[0]

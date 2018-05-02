import numpy as np

from policy.policy import profit


def create_policies(demand_distribution, product_cost, fixed_order_cost, holding_cost_per_interval,
                  max_stock, selling_price_low, selling_price_high, threshold=1e-5):
    """
    Uses a known (or predicted) demand distribution and the Bellman equation to determine an optimal ordering and pricing policy
    """
    expected_profits = np.zeros(max_stock + 1)
    remaining_stock = np.arange(max_stock + 1).reshape((-1, 1, 1, 1))
    order_quantity = np.arange(max_stock + 1).reshape((1, -1, 1, 1))
    selling_prices = np.arange(selling_price_low, selling_price_high + 1).reshape((1, 1, -1, 1))
    demand = np.arange(max_stock + 1).reshape((1, 1, 1, -1))
    probabilities = demand_distribution(demand, selling_prices)
    sales = np.minimum(demand, remaining_stock)
    all_expected_profits = None
    difference = float('inf')
    iteration = 0

    # Stop iterating if changes in expected profit are below the threshold
    while difference > threshold:
        all_expected_profits = np.sum(probabilities * (
                profit(remaining_stock, sales, order_quantity, selling_prices, product_cost, fixed_order_cost,
                       holding_cost_per_interval)
                + iteration * expected_profits[np.clip(remaining_stock + order_quantity - sales, 0, max_stock)]
        ), axis=3) / (iteration + 1)
        old_expected_profits = expected_profits
        expected_profits = np.max(all_expected_profits, axis=(1, 2))
        difference = np.max(np.absolute(expected_profits - old_expected_profits))
        iteration += 1

    # Combine order_quantity and price dimension, because we cannot get argmax over multiple dimensions
    policy = np.argmax(all_expected_profits.reshape(len(remaining_stock), -1), axis=1)
    order_policy_indices, price_policy_indices = np.unravel_index(policy, (order_quantity.shape[1], selling_prices.shape[2]))
    order_policy = order_quantity[0,:,0,0][order_policy_indices]
    pricing_policy = selling_prices[0,0,:,0][price_policy_indices]

    print('Ordering policy:')
    print(order_policy)
    print('Pricing policy:')
    print(pricing_policy)

    def order_policy_function(stock):
        return int(order_policy[np.clip(stock, 0, len(order_policy) - 1)])

    def pricing_policy_function(stock):
        return int(pricing_policy[np.clip(stock, 0, len(pricing_policy) - 1)])
    
    return order_policy_function, pricing_policy_function

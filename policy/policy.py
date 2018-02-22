import numpy as np


def create_policy(demand_distribution, product_cost, fixed_order_cost, holding_cost_per_interval,
                  max_stock, selling_price_low, selling_price_high, max_iterations=10):
    # TODO: don't use selling price from outside
    # TODO: don't use max_stock from outside
    remaining_stock = np.arange(0, max_stock + 1).reshape((-1, 1, 1, 1))
    order_quantity = np.arange(0, max_stock + 1).reshape((1, -1, 1, 1))
    selling_prices = np.arange(selling_price_low, selling_price_high).reshape((1, 1, -1, 1))
    demand = np.arange(0, max_stock + 1).reshape((1, 1, 1, -1))

    expected_profits = np.zeros(len(remaining_stock))
    order_policy = np.zeros(len(remaining_stock))
    price_policy = np.zeros(len(remaining_stock))
    # TODO: use selling price

    for _ in range(max_iterations):
        old_order_policy = order_policy
        order_policy, price_policy, expected_profits = bellman_equation(demand_distribution, product_cost, fixed_order_cost,
                                                          holding_cost_per_interval, selling_prices, expected_profits,
                                                          remaining_stock, order_quantity, demand, iterations=100)
        print(order_policy)
        print(price_policy)
        # ignore non-orders, i.e. orders of zero products for the minimum order
        min_order = np.min(order_policy[order_policy > 0])
        max_order = np.max(order_policy)

        lower_limit = order_quantity[0, 1, 0, 0]
        upper_limit = order_quantity[0, -1, 0, 0]

        if min_order == lower_limit:
            order_start = min_order // 2
        else:
            # Add some unused order quantities as option
            # TODO: use something that depends on min_order size (e.g. min_order * 10%)
            order_start = max(min_order - 3, 1)

        if max_order == upper_limit:
            order_end = max_order * 2
        else:
            # TODO: see order_start
            order_end = max_order + 3

        # The array must always contain a zero so that the merchant is able to not order product.
        # Make some space in the array and set the first element to zero.
        order_quantity = np.arange(order_start - 1, order_end + 1).reshape((1, -1, 1, 1))
        order_quantity[0, 0, 0, 0] = 0

        if np.array_equal(order_policy, old_order_policy):
            # The policy has converged
            break

    return order_policy, price_policy


def bellman_equation(demand_distribution, product_cost, fixed_order_cost, holding_cost_per_interval, selling_prices,
                     expected_profits, remaining_stock, order_quantity, demand, iterations):
    # TODO: remove dimension magic numbers
    probabilities = demand_distribution(demand, selling_prices)
    sales = np.minimum(demand, remaining_stock + order_quantity)

    for i in range(1, iterations + 1):
        all_expected_profits = np.sum(probabilities * (
                profit(remaining_stock, sales, order_quantity, selling_prices, product_cost, fixed_order_cost,
                       holding_cost_per_interval)
                + i * expected_profits[np.minimum(remaining_stock + order_quantity - sales, len(expected_profits) - 1)]
        ), axis=3) / (i + 1)
        expected_profits = np.max(all_expected_profits, axis=(1, 2))

    # Combine order_quantity and price dimension, because we cannot get argmax over multiple dimensions
    policy = np.argmax(all_expected_profits.reshape(remaining_stock.shape[0], -1), axis=1)
    order_policy_indices, price_policy_indices = np.unravel_index(policy, (remaining_stock.shape[0], selling_prices.shape[2]))
    #order_policy_indices = np.argmax(all_expected_profits, axis=1)
    order_policy = order_quantity[0, :, 0, 0][order_policy_indices]
    price_policy = selling_prices[0, 0, :, 0][price_policy_indices]
    return order_policy, price_policy, expected_profits


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
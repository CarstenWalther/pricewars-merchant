import numpy as np


def get_demand_estimation(waiting_time_distribution, interval_length, array_length, iterations=10000):
    # interval_length in seconds
    event_counter = np.zeros(array_length)
    interval_time = 0.0
    event_count = 0
    for _ in range(iterations):
        interval_time += waiting_time_distribution()
        while interval_time >= interval_length:
            event_counter[min(event_count, array_length-1)] += 1
            event_count = 0
            interval_time -= interval_length
        event_count += 1

    return event_counter / np.sum(event_counter)


def waiting_time_distribution():
    return np.random.exponential(scale=1.0)


interval_length = 5.0
array_length = 20
demand_distribution = get_demand_estimation(waiting_time_distribution, interval_length, array_length)
print(demand_distribution)
print(np.sum(demand_distribution))
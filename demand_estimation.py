import numpy as np


def estimate_demand_distribution(waiting_time_distribution, interval_length_in_seconds, array_length, iterations=10000):
    event_counter = np.zeros(array_length)
    interval_time = 0.0
    event_count = 0
    for _ in range(iterations):
        interval_time += waiting_time_distribution()
        while interval_time >= interval_length_in_seconds:
            event_counter[min(event_count, array_length-1)] += 1
            event_count = 0
            interval_time -= interval_length_in_seconds
        event_count += 1

    return event_counter / np.sum(event_counter)

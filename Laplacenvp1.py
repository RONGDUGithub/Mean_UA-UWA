import numpy as np


def laplace_noisy(epsilon):
    n_value = np.random.laplace(0, 2 / epsilon, 1)
    return n_value


def laplace_mech(data, epsilon):
    for i in range(len(data)):
        data[i] += laplace_noisy(epsilon)
    return data


def linear_normalize(array, min_val, max_val):
    normalized_array = (array - min_val) * 2 / (max_val - min_val)-1
    return normalized_array


def linear_denormalize(normalized_array, min_val, max_val):
    denormalized_array = 0.5 * (normalized_array + 1) * (max_val - min_val) + min_val
    return denormalized_array


def laplace1(data, min_val, max_val, epsilon):
    data[data > max_val] = max_val
    data[data < min_val] = min_val
    normalized_data = linear_normalize(data, min_val, max_val)
    # print("epsilon:", epsilon)
    # print("normalized_data:", normalized_data)
    data_noisy = laplace_mech(normalized_data, epsilon)
    # print("data_noisy:", data_noisy)
    data_noisy_denor = linear_denormalize(data_noisy, min_val, max_val)
    return data_noisy_denor, data_noisy

import numpy as np
import random
import scipy
from numpy import linalg as LA
from pm_backup import discretize_interval_pm, discretize_probability_pm
from duchi_backup import divide_interval_and_boundaries_sr, discretize_probability_sr
from sw import EMS, EM


def duchi_distribution(result_duchi, eps, domain_bins=1024):
    epsilon_sr = eps
    d_sr = domain_bins
    db_sr = 2
    n = domain_bins

    mid_values__minus1_to_1_sr, BM_sr = divide_interval_and_boundaries_sr(d_sr)
    # noise_discrete_sr = SR_mechanism(user_value, epsilon_sr)

    transform = np.zeros((db_sr, d_sr))
    for i in range(len(mid_values__minus1_to_1_sr)):
        t = mid_values__minus1_to_1_sr[i]
        probabilities = discretize_probability_sr(t, epsilon_sr)
        transform[:, i] = probabilities

    srplus = (np.exp(epsilon_sr) + 1) / (np.exp(epsilon_sr) - 1)
    srminus = -(np.exp(epsilon_sr) + 1) / (np.exp(epsilon_sr) - 1)

    user_value_noise_sr = result_duchi
    H_sr = np.zeros(2)
    H_sr[0] = np.sum(user_value_noise_sr > 0)
    H_sr[1] = np.sum(user_value_noise_sr < 0)

    location = [None] * len(result_duchi)  # 初始化 location 列表

    for i in range(len(result_duchi)):
        if user_value_noise_sr[i] > 0:
            location[i] = 0
        else:
            location[i] = 1

    max_iteration = 10000
    loglikelihood_threshold = 1e-3

    #?????
    # ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-C, C))
    # return EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    ns_hist = H_sr

    return location, transform, EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(result_duchi)

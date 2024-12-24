import numpy as np
import random
import scipy
from numpy import linalg as LA
from pm_backup import discretize_interval_pm, discretize_probability_pm
from sw import EMS
from all_backup import *


def pm_distribution(result_piecewise, eps, randomized_bins=1024, domain_bins=1024):
    noisy_samples = result_piecewise
    ee = np.exp(eps)
    ee2 = np.exp(eps/2)

    C = (ee2 + 1) / (ee2 - 1)
    p_pm = (ee-ee2)/(2*ee2+2)
    q_pm = p_pm/ee

    test = p_pm*(C-1)+q_pm*(1+C)

    midpoints, BM_pm, midpoints_minus1_to_1_pm = discretize_interval_pm(C, domain_bins)

    # report matrix
    m = randomized_bins
    n = domain_bins

    transform = np.ones((m, n))

    for i, t in enumerate(midpoints_minus1_to_1_pm):
        probabilities = discretize_probability_pm(C, t, p_pm, q_pm, randomized_bins, BM_pm)
        transform[:, i] = probabilities


    max_iteration = 10000
    loglikelihood_threshold = 1e-3

    #?????
    ns_hist, _ = np.histogram(result_piecewise, bins=randomized_bins, range=(-C, C))
    # return EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    # test = np.sum(ns_hist)
    location = get_positions(-C, C, noisy_samples)
    return location, transform, ns_hist, EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(result_piecewise)

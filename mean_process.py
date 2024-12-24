import numpy as np
import matplotlib.pyplot as plt
from Duchinvp import duchi
from Hybridnvp import hybrid
from Laplacenvp import laplace1
from Piecewisenvp import piecewise
from SWunbiasednvp import sw_unbiased
from sw import sw
from discrete import *
from bayesain_updating import *
from calculate_matrix import *
from variance_final import *
import pandas as pd

user_number = 1000
data = np.random.beta(2, 5, user_number)


min_data = 0
max_data = 1
num_runs = 2
epsilons = [0.5, 1.0, 1.5, 2.0, 2.5]

all_mean_aggre = []
all_mean_laplace = []
all_mean_duchi = []
all_mean_pm = []
all_mean_baseline = []
all_mean_sw = []

for epsilon in epsilons:
    epsilon_default = epsilon/4

    for run in range(num_runs):
        result_duchi, noise_duchi = duchi(data, min_data, max_data, epsilon_default)
        result_piecewise, noise_piecewise = piecewise(data, min_data, max_data, epsilon_default)
        result_laplace, noise_laplace = laplace1(data, min_data, max_data, epsilon_default)
        result_sw_unbiased, result_sw_biased = sw_unbiased(data, min_data, max_data, epsilon_default)

        d_default = 100
        final = np.ones(d_default) / d_default
        midpoints_minus1_to_1 = np.linspace(-1, 1 - 2 / d_default, d_default) + 1 / d_default
        midpoints_minus0_to_1 = np.linspace(0, 1 - 1 / d_default, d_default) + 1 / (2 * d_default)

        mean_final = [0] * user_number
        for i in range(1, user_number + 1):
            final = np.ones(d_default) / d_default
            epsilon_duchi = epsilon_default
            final = duchi_bayes(noise_duchi[i - 1], midpoints_minus1_to_1, epsilon_duchi, final)

            epsilon_lap = epsilon_default
            final = laplace_bayes(d_default, epsilon_lap, noise_laplace[i - 1], final)

            epsilon_pm = epsilon_default
            t = noise_piecewise[i - 1]
            final = pm_pre_bayes(t, epsilon_pm, d_default, final)

            epsilon_sw = epsilon_default
            t = result_sw_biased[i - 1]
            final = sw_pre_bayes(t, epsilon_sw, d_default, final)
            normalized_value = 2 * (data[i - 1] - min_data) / (max_data - min_data) - 1

            var_duchi = compute_duchi_variance(final, epsilon_duchi, d_default, midpoints_minus1_to_1)
            var_pm = compute_pm_variance(final, epsilon_pm, d_default, midpoints_minus1_to_1)
            var_laplace = compute_laplace_variance(final, epsilon_lap, d_default, midpoints_minus1_to_1)
            var_sw_unbiased = compute_sw_variance(final, epsilon_sw, d_default, midpoints_minus1_to_1)
            var_all = 1/var_duchi + 1/var_pm + 1/var_laplace + 1/var_sw_unbiased
            w_duchi = 1/var_duchi/var_all
            w_pm = 1/var_pm/var_all
            w_laplace = 1/var_laplace/var_all
            w_sw_unbiased = 1/var_sw_unbiased/var_all

            mean_final[i - 1] = (
                    w_duchi * result_duchi[i - 1] +
                    w_pm * result_piecewise[i - 1] +
                    w_laplace * result_laplace[i - 1] +
                    w_sw_unbiased * result_sw_unbiased[i - 1]
            )

        mean_duchi = np.mean(result_duchi)
        mean_piecewise = np.mean(result_piecewise)
        mean_laplace = np.mean(result_laplace)
        mean_sw = np.mean(result_sw_unbiased)
        mean_baseline = np.mean([mean_duchi, mean_piecewise, mean_laplace, mean_sw])
        mean_aggre = np.mean(mean_final)

        all_mean_laplace.append(mean_laplace)
        all_mean_duchi.append(mean_duchi)
        all_mean_pm.append(mean_piecewise)
        all_mean_sw.append(mean_sw)
        all_mean_baseline.append(mean_baseline)
        all_mean_aggre.append(mean_aggre)

mean_ground_truth = np.mean(data)
MSE_laplace = np.mean((np.array(all_mean_laplace) - mean_ground_truth)**2)
MSE_duchi = np.mean((np.array(all_mean_duchi) - mean_ground_truth)**2)
MSE_pm = np.mean((np.array(all_mean_pm) - mean_ground_truth)**2)
MSE_sw = np.mean((np.array(all_mean_sw) - mean_ground_truth)**2)
MSE_baseline = np.mean((np.array(all_mean_baseline) - mean_ground_truth)**2)
MSE_aggre = np.mean((np.array(all_mean_aggre) - mean_ground_truth)**2)

all_mses = np.array([MSE_laplace, MSE_duchi, MSE_pm, MSE_sw, MSE_baseline, MSE_aggre])

# 将结果写入 result.txt 文件
with open('result.txt', 'w') as f:
    f.write("All MSE values:\n")
    f.write(str(all_mses))
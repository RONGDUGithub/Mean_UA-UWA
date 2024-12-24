import numpy as np
import pandas as pd

# 导入您提及的模块（确保这些模块在您的环境中已经安装和可用）
from Duchinvp import duchi
from Hybridnvp import hybrid
from Laplacenvp import laplace
from Piecewisenvp import piecewise
from SWunbiasednvp import sw_unbiased
from entropyfile import calculate_entropy, calculate_weights

def calculate_means_and_entropies(data, privacy_budget):
    min_data = min(data)
    max_data = max(data)

    # 调用不同的函数来获取扰动后的数据集
    result_duchi = duchi(data, min_data, max_data, privacy_budget)
    result_piecewise = piecewise(data, min_data, max_data, privacy_budget)
    result_hybrid = hybrid(data, min_data, max_data, privacy_budget)
    result_laplace = laplace(data, min_data, max_data, privacy_budget)
    result_sw_unbiased = sw_unbiased(data, min_data, max_data, privacy_budget)

    # 计算均值
    mean_groundtruth = np.mean(data, axis=0)
    mean_duchi = np.mean(result_duchi, axis=0)
    mean_piecewise = np.mean(result_piecewise, axis=0)
    mean_hybrid = np.mean(result_hybrid, axis=0)
    mean_laplace = np.mean(result_laplace, axis=0)
    mean_sw_unbiased = np.mean(result_sw_unbiased, axis=0)
    mean_vector = [mean_duchi, mean_piecewise, mean_hybrid, mean_laplace, mean_sw_unbiased]

    print("均值：", mean_groundtruth, mean_duchi, mean_piecewise, mean_hybrid, mean_laplace, mean_sw_unbiased)

    # 计算熵
    entropy_duchi = calculate_entropy(result_duchi)
    entropy_piecewise = calculate_entropy(result_piecewise)
    entropy_hybrid = calculate_entropy(result_hybrid)
    entropy_laplace = calculate_entropy(result_laplace)
    entropy_sw_unbiased = calculate_entropy(result_sw_unbiased)

    entropy_vector = [entropy_duchi, entropy_piecewise, entropy_hybrid, entropy_laplace, entropy_sw_unbiased]
    print("熵值：", entropy_duchi, entropy_piecewise, entropy_hybrid, entropy_laplace, entropy_sw_unbiased)

    # 检查熵值是否有NaN，并计算加权平均值
    if np.isnan(entropy_vector).any():
        print("存在无效值（NaN）")
    else:
        print("测试")
        weights = calculate_weights(entropy_vector)
        print("weights", weights)
        # 计算加权均值
        # weighted_mean = calculate_weighted_mean(vectors, weights)
        # 计算加权均值
        weighted_average = np.average(mean_vector, weights=weights)

        # weights = calculate_weights(entropy_vector)

        # weighted_average = np.average(mean_vector[1:], weights=weights)  # 排除真实均值，因为我们的目标是加权其他估计值
        print("weighted_average", weighted_average)
    mean_vector2 = [mean_duchi, mean_piecewise, mean_hybrid, mean_laplace, mean_sw_unbiased, weighted_average]
    print("mean_vector2", mean_vector2)
    # 返回均值和熵
    return mean_groundtruth, mean_vector2

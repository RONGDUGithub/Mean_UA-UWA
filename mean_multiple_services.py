import time
from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from Duchinvp1 import duchi
from Hybridnvp import hybrid
from Laplacenvp1 import laplace1
from Piecewisenvp1 import piecewise
from SWunbiasednvp1 import sw_unbiased
from sw import sw
from discrete import *
from bayesain_updating import *
from calculate_matrix import *
from variance_final import *



def multi_mechanism_process(mechanisms, data, min_data, max_data, epsilons, use_cuda=True):
    """
    处理多个机制的函数，类似multi_mechanism_switch的结构
    """
    denor_results = []      # 机制的主要结果
    noise_data = []   # 噪声数据
    means = []        # 平均值

    # 检查epsilons是否匹配mechanisms
    if isinstance(epsilons, list):
        if len(epsilons) != len(mechanisms):
            raise ValueError("Number of epsilons must match number of mechanisms")
        epsilon_dict = {i: eps for i, eps in enumerate(epsilons)}
    else:
        epsilon_dict = epsilons

    for mech_idx, mechanism in enumerate(mechanisms):
        current_epsilon = epsilon_dict[mech_idx]

        if mechanism.lower() == 'duchi':
            result, noise = duchi(data, min_data, max_data, current_epsilon)
            mean = np.mean(result)
            if use_cuda:
                result = torch.tensor(result).cuda()
                noise = torch.tensor(noise).cuda()

            denor_results.append(result)
            noise_data.append(noise)
            means.append(mean)

        elif mechanism.lower() == 'laplace':
            result, noise = laplace1(data, min_data, max_data, current_epsilon)
            mean = np.mean(result)
            if use_cuda:
                result = torch.tensor(result).cuda()
                noise = torch.tensor(noise).cuda()

            denor_results.append(result)
            noise_data.append(noise)
            means.append(mean)

        elif mechanism.lower() == 'piecewise':
            result, noise = piecewise(data, min_data, max_data, current_epsilon)
            mean = np.mean(result)
            if use_cuda:
                result = torch.tensor(result).cuda()
                noise = torch.tensor(noise).cuda()

            denor_results.append(result)
            noise_data.append(noise)
            means.append(mean)

        elif mechanism.lower() == 'sw':
            result_unbiased, result_biased = sw_unbiased(data, min_data, max_data, current_epsilon)
            mean = np.mean(result_unbiased)
            if use_cuda:
                result_unbiased = torch.tensor(result_unbiased).cuda()
                result_biased = torch.tensor(result_biased).cuda()

            denor_results.append(result_unbiased)
            noise_data.append(result_biased)
            means.append(mean)

        else:
            raise ValueError("Mechanism must be one of: 'duchi', 'laplace', 'piecewise', 'sw'")

    return denor_results, noise_data, means


def compute_weighted_mean(mechanisms, epsilons, batch_size, denor_results, noise_data, d_default, midpoints_minus1_to_1):

    # 创建epsilon字典
    eps_dict = {mech: eps for mech, eps in zip(mechanisms, epsilons)}
    results = {}
    final = torch.full((d_default,), 1 / d_default).repeat(batch_size, 1).cuda()

    for i, mech in enumerate(mechanisms):
        # 使用列索引来获取对应机制的噪声
        noise = noise_data[i, :]  # [batch_size]
        eps = eps_dict[mech]
        if mech == 'duchi':
            final = duchi_bayes_batch(noise, midpoints_minus1_to_1, eps, final.clone())
        elif mech == 'piecewise':
            final = pm_pre_bayes(noise, eps, d_default, final.clone())
        elif mech == 'laplace':
            final = laplace_bayes(d_default, eps, noise, final.clone())
        elif mech == 'sw':
            final = sw_pre_bayes(noise, eps, d_default, final.clone())


    # 计算方差
    variances = {}
    for mech, eps in eps_dict.items():
        eps = torch.tensor(eps, device='cuda')
        if mech == 'duchi':
            variances[mech] = compute_duchi_variance(final, eps, d_default, midpoints_minus1_to_1)
        elif mech == 'piecewise':
            variances[mech] = compute_pm_variance(final, eps, d_default, midpoints_minus1_to_1)
        elif mech == 'laplace':
            variances[mech] = compute_laplace_variance(final, eps, d_default, midpoints_minus1_to_1)
        elif mech == 'sw':
            variances[mech] = compute_sw_variance(final, eps, d_default, midpoints_minus1_to_1)
    # print('variances', variances)

    # 计算总方差
    first_var = next(iter(variances.values()))
    var_all = torch.zeros_like(first_var, dtype=torch.float64).cuda()
    for var in variances.values():
        var_all += 1 / var

    # 计算权重
    weights = {mech: (1 / var) / var_all for mech, var in variances.items()}
    weighted_sum = torch.zeros(batch_size, device='cuda')  # 如果需要在GPU上

    for g, mech in enumerate(mechanisms):
        weighted_sum += weights[mech] * denor_results[g]


    return torch.sum(weighted_sum)

def compute_weighted_mean1(mechanisms, epsilons, batch_size, denor_results, noise_data, d_default, midpoints_minus1_to_1):
    final = torch.full((d_default,), 1 / d_default).repeat(batch_size, 1).cuda()

    # 贝叶斯更新
    for i, (mech, eps) in enumerate(zip(mechanisms, epsilons)):
        noise = noise_data[i, :]  # [batch_size]
        if mech == 'duchi':
            final = duchi_bayes_batch(noise, midpoints_minus1_to_1, eps, final.clone())
        elif mech == 'piecewise':
            final = pm_pre_bayes(noise, eps, d_default, final.clone())
        elif mech == 'laplace':
            final = laplace_bayes(d_default, eps, noise, final.clone())
        elif mech == 'sw':
            final = sw_pre_bayes(noise, eps, d_default, final.clone())


    # 计算方差
    variances = []
    for i, (mech, eps) in enumerate(zip(mechanisms, epsilons)):
        eps = torch.tensor(eps, device='cuda')
        if mech == 'duchi':
            variance = compute_duchi_variance(final, eps, d_default, midpoints_minus1_to_1)
        elif mech == 'piecewise':
            variance = compute_pm_variance(final, eps, d_default, midpoints_minus1_to_1)
        elif mech == 'laplace':
            variance = compute_laplace_variance(final, eps, d_default, midpoints_minus1_to_1)
        elif mech == 'sw':
            variance = compute_sw_variance(final, eps, d_default, midpoints_minus1_to_1)
        variances.append(variance)

    # 计算总方差
    var_all = torch.zeros_like(variances[0], dtype=torch.float64).cuda()
    for var in variances:
        var_all += 1 / var

    # 计算权重
    weights = [(1 / var) / var_all for var in variances]
    weighted_sum = torch.zeros(batch_size, device='cuda')

    # 应用权重
    for g in range(len(mechanisms)):
        weighted_sum += weights[g] * denor_results[g]

    return torch.sum(weighted_sum)


def compute_weighted_mean2(mechanisms, epsilons, batch_size, batch_data0, noise_data, d_default, midpoints_minus1_to_1):
    final = torch.full((d_default,), 1 / d_default).repeat(batch_size, 1).cuda()

    # 贝叶斯更新
    for i, (mech, eps) in enumerate(zip(mechanisms, epsilons)):
        noise = noise_data[i, :]  # [batch_size]
        if mech == 'duchi':
            final = duchi_bayes_batch(noise, midpoints_minus1_to_1, eps, final.clone())
        elif mech == 'piecewise':
            final = pm_pre_bayes(noise, eps, d_default, final.clone())
        elif mech == 'laplace':
            final = laplace_bayes(d_default, eps, noise, final.clone())
        elif mech == 'sw':
            final = sw_pre_bayes(noise, eps, d_default, final.clone())
    # print(sum(final))
        # 找出真实的bucket位置
    true_positions = find_bucket(batch_data0.squeeze(), d_default)  # [batch_size]

    #     # 取出对应位置的概率值
    # batch_indices = torch.arange(batch_size).cuda()
    # true_probabilities = final[batch_indices, true_positions]  # [batch_size]
    score = calculate_mse(true_positions, final)
    return score


def calculate_mse(true_positions, predictions):
    """
    计算批量数据的加权均方误差 (向量化实现)

    参数:
    true_positions: shape为(batch_size,)的张量，每个元素是真实位置
    predictions: shape为(batch_size, sequence_length)的张量，
                每行表示一个样本在各个位置的预测概率分布

    返回:
    mse: 批量数据的均方误差
    """
    # 创建位置索引向量并移动到正确的设备
    positions = torch.arange(predictions.shape[1],
                             device=predictions.device,
                             dtype=predictions.dtype)

    # 确保 true_positions 在正确的设备上
    true_positions = true_positions.to(predictions.device)

    # 扩展 true_positions 以便广播
    true_positions = true_positions.unsqueeze(1)

    # 计算每个位置的平方误差
    position_errors = (positions - true_positions) ** 2  # 这里改用平方误差

    # 将误差与概率相乘并求和
    weighted_errors = (predictions * position_errors).sum(dim=1)

    # 计算平均误差
    mse = weighted_errors.mean()

    return mse

def calculate_mae(true_positions, predictions):
    positions = torch.arange(predictions.shape[1],
                             device=predictions.device,
                             dtype=predictions.dtype)  # 确保数据类型一致

    # 确保 true_positions 在正确的设备上
    true_positions = true_positions.to(predictions.device)

    # 扩展 true_positions 以便广播
    true_positions = true_positions.unsqueeze(1)

    # 计算每个位置的绝对误差
    position_errors = torch.abs(positions - true_positions)

    # 将误差与概率相乘并求和
    weighted_errors = (predictions * position_errors).sum(dim=1)

    # 计算平均误差
    mae = weighted_errors.mean()
    return mae

def sample_data(input_file, n_cols):
    # 读取数据
    df = pd.read_csv('output4.csv', header=None)
    total_rows = len(df)
    samples_per_col = total_rows // n_cols  # 每列需要采样的数量

    print(f"总行数: {total_rows}")
    print(f"每列采样数量: {samples_per_col}")

    # 只选择指定列数
    df = df.iloc[:, :n_cols]

    # 创建结果DataFrame
    result_df = pd.DataFrame().reindex_like(df)

    # 保持所有可用的行索引
    available_indices = set(range(total_rows))

    # 对每一列进行采样
    for col in df.columns:
        # 检查是否还有足够的可用行
        if len(available_indices) < samples_per_col:
            print(f"警告：列 {col} 没有足够的可用行进行采样")
            current_indices = np.random.choice(list(available_indices),
                                               size=len(available_indices),
                                               replace=False)
        else:
            current_indices = np.random.choice(list(available_indices),
                                               size=samples_per_col,
                                               replace=False)

        # 将采样到的数据放入结果DataFrame
        result_df.iloc[list(current_indices), col] = df.iloc[list(current_indices), col]

        # 从可用索引中移除已使用的索引
        available_indices -= set(current_indices)

        # 打印剩余可用行数
        print(f"列 {col} 采样后，剩余可用行数: {len(available_indices)}")

    return result_df

def find_bucket(data, num_buckets):
    """
    将[-1,1]区间均分为num_buckets份，找到data所在的bucket
    Args:
        data: 输入数据 (假设已经归一化到[-1,1])
        num_buckets: bucket数量
    Returns:
        bucket索引
    """
    bucket_size = 2.0 / num_buckets  # 每个bucket的大小
    # 将[-1,1]映射到[0,num_buckets]
    bucket_index = ((data + 1) / bucket_size).long()
    # 处理边界情况
    bucket_index = torch.clamp(bucket_index, 0, num_buckets-1)
    return bucket_index
# # 生成示例数据
# data0 = np.random.normal(loc=0, scale=1, size=1000)  # 生成1000个正态分布的数据点
# min_data0 = np.min(data0)
# max_data0 = np.max(data0)
#
#
# # 定义机制和对应的epsilon值
# mechanisms = ['duchi', 'laplace', 'piecewise', 'sw']
# epsilons = [0.1, 0.1, 0.1, 0.1]  # 对应每个机制的epsilon值
#
# #
# # # 调用函数
# # results, user_noise, mean_list = multi_mechanism_process(mechanisms, data0, min_data0, max_data0, epsilons)
# #
# # print(results)
# # print(user_noise)
# # print(mean_list)
#

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace
import math
from discrete import *
from calculate_matrix import *

def duchi_bayes_batch(user_value_noise_sr, mid_values__minus1_to_1_sr, epsilon, final):
    # 将 epsilon 转换为 tensor（如果不是的话）
    epsilon = torch.tensor(epsilon, device=user_value_noise_sr.device, dtype=torch.float32)

    # 预先计算 exp(epsilon) 以避免重复计算
    exp_epsilon = torch.exp(epsilon)

    # 扩展维度以进行广播
    value = mid_values__minus1_to_1_sr.unsqueeze(0)  # [1, d_default]
    user_value_noise_sr = user_value_noise_sr.unsqueeze(1)  # [batch_size, 1]

    # 计算条件概率
    term1 = torch.where(user_value_noise_sr > 0,
                        (((exp_epsilon - 1) * value) / (2 * exp_epsilon + 2)) + 1 / 2,
                        1 - (((exp_epsilon - 1) * value) / (2 * exp_epsilon + 2) + 1 / 2))

    # 元素乘法
    term2 = term1 * final

    # 对每个batch求和
    term3 = torch.sum(term2, dim=1)

    # 归一化
    final = term2 / term3.reshape(-1, 1)

    return final


def duchi_bayes(user_value_noise_sr, mid_values__minus1_to_1_sr, epsilon, final):
    d_sr = len(mid_values__minus1_to_1_sr)
    value = mid_values__minus1_to_1_sr
    term1 = torch.where(user_value_noise_sr > 0,
                        (((torch.exp(epsilon) - 1) * value) / (2 * torch.exp(epsilon) + 2)) + 1 / 2,
                        1 - (((torch.exp(epsilon) - 1) * value) / (2 * torch.exp(epsilon) + 2) + 1 / 2))
    term2 = term1 * final
    term3 = torch.sum(term2)
    final = term2 / term3
    return final


def pm_bayes(d_pm, matrix_pm, nearest_index, final):

    term1 = matrix_pm[nearest_index, :]
    term2 = term1 * final
    # term3 = torch.sum(term2)
    term3 = torch.sum(term2, dim=-1, keepdim=True)
    final = term2 / term3
    return final


def sw_bayes(d_sw, matrix_sw, nearest_index, final):
    term1 = matrix_sw[nearest_index, :]
    term2 = term1 * final
    # term3 = torch.sum(term2)
    term3 = torch.sum(term2, dim=-1, keepdim=True)
    final = term2 / term3
    return final


def laplace_bayes(d_lap, epsilon_lap, v_perturbed, final):
    # 确保 epsilon_lap 是 tensor
    if not isinstance(epsilon_lap, torch.Tensor):
        epsilon_lap = torch.tensor(epsilon_lap, device='cuda', dtype=torch.float)

    # 初始化变量
    term2 = torch.zeros(d_lap, device='cuda')
    midpoints = torch.linspace(-1, 1 - 2 / d_lap, d_lap, device='cuda') + 1 / d_lap

    # 计算 scale parameter
    c = torch.tensor(2.0, device='cuda') / epsilon_lap  # 确保是 tensor

    # 计算百分位数
    c_np = c.item()  # 转换为 Python float 用于 scipy.stats.laplace
    percentile_10 = laplace.ppf(0.10, scale=c_np)
    percentile_90 = laplace.ppf(0.90, scale=c_np)

    # 转换回 tensor
    percentile_10 = torch.tensor(percentile_10, device='cuda')
    percentile_90 = torch.tensor(percentile_90, device='cuda')

    # 计算过程
    v_perturbed = torch.clamp(v_perturbed, percentile_10, percentile_90)
    abs_v_perturbed = torch.abs(v_perturbed)
    ol = torch.sqrt(abs_v_perturbed)
    a = v_perturbed - ol  # lower bound of integration
    b = v_perturbed + ol  # upper bound of integration

    # 生成积分点
    t = torch.linspace(0, 1, 100, device='cuda').unsqueeze(0).expand(len(a), -1)

    # 计算截断拉普拉斯PDF和
    result = compute_truncated_laplace_pdf_sum(a, b, midpoints, c, num_points=100)
    result = result / torch.sum(result, dim=1).reshape(-1, 1)

    # 最终计算
    term2 = result * final
    term3 = torch.sum(term2, dim=1)
    final = term2 / term3.reshape(-1, 1)

    return final

def compute_truncated_laplace_pdf_sum(a, b, mu, c, num_points=100):
    # a, b: [batch_size]
    # mu: [d_default]
    # c: scalar
    batch_size = a.size(0)
    num_midpoints = mu.size(0)

    # 为每个batch生成积分点
    # [batch_size, num_points]
    t = torch.linspace(0, 1, num_points, device='cuda').unsqueeze(0).expand(batch_size, -1)
    x = a.unsqueeze(1) + t * (b - a).unsqueeze(1)  # [batch_size, num_points]

    # 计算dx
    dx = ((b - a) / num_points).unsqueeze(1)  # [batch_size, 1]

    # 调整维度以便广播
    x = x.unsqueeze(2)  # [batch_size, num_points, 1]
    mu = mu.unsqueeze(0).unsqueeze(1)  # [1, 1, d_default]

    # 计算PDF值
    pdf_values = torch.exp(-torch.abs(x - mu) / c) / (2 * c)  # [batch_size, num_points, d_default]

    # 对每个batch和每个midpoint进行积分
    result = torch.sum(pdf_values * dx.unsqueeze(2), dim=1)  # [batch_size, d_default]

    return result


def pm_pre_bayes(t, epsilon_pm, d_default, final):
    # 确保 epsilon_pm 是 tensor 并且在正确的设备上
    if not isinstance(epsilon_pm, torch.Tensor):
        epsilon_pm = torch.tensor(epsilon_pm, device='cuda', dtype=torch.float)

    # 计算常数C
    C = (torch.exp(epsilon_pm / 2) + 1) / (torch.exp(epsilon_pm / 2) - 1)

    # 离散化区间
    midpoints_pm, BM_pm, midpoints_minus1_to_1_pm, db_pm = discretize_interval_pm(C.item(), d_default)

    # 转换到CUDA
    midpoints_minus1_to_1_pm = torch.tensor(midpoints_minus1_to_1_pm, device='cuda')
    db_pm = torch.tensor(db_pm, device='cuda')
    BM_pm = torch.tensor(BM_pm, device='cuda')

    # 计算矩阵
    matrix_pm = compute_matrix_pm(epsilon_pm.item(),
                                  midpoints_minus1_to_1_pm.cpu().numpy(),
                                  db_pm.cpu().numpy(),
                                  BM_pm.cpu().numpy())
    matrix_pm = torch.tensor(matrix_pm, device='cuda')

    # 批处理计算
    midpoints_expanded = midpoints_minus1_to_1_pm.unsqueeze(0).expand(t.size(0), -1)  # [batch_size, d_default]
    t_expanded = t.unsqueeze(1)  # [batch_size, 1]
    nearest_index_pm = torch.argmin(torch.abs(midpoints_expanded - t_expanded), dim=1)  # [batch_size]

    # 贝叶斯更新
    final = pm_bayes(d_default, matrix_pm, nearest_index_pm, final)
    return final


def sw_pre_bayes(t, epsilon_sw, d_default, final):
    # t: [batch_size]
    # epsilon_sw: scalar
    # d_default: int
    # final: [batch_size, d_default]

    # 确保 epsilon_sw 是 tensor
    if not isinstance(epsilon_sw, torch.Tensor):
        epsilon_sw = torch.tensor(epsilon_sw, device='cuda', dtype=torch.float)

    # 计算b_sw
    exp_eps = torch.exp(epsilon_sw)
    b_sw = ((epsilon_sw * exp_eps - exp_eps + 1) /
            (2 * exp_eps * (exp_eps - 1 - epsilon_sw)))

    # 离散化区间
    midpoints_sw, BM_sw, midpoints_minus0_to_1plusb, db_sw = discretize_interval_sw(b_sw.item(), d_default)

    # 转换到CUDA
    midpoints_minus0_to_1plusb = torch.tensor(midpoints_minus0_to_1plusb, device='cuda')  # [d_default]

    # 计算转换矩阵
    matrix_sw = sw_transform(epsilon_sw.item(), d_default)
    matrix_sw = torch.tensor(matrix_sw, device='cuda')

    # 调整维度以进行批处理计算
    midpoints_expanded = midpoints_minus0_to_1plusb.unsqueeze(0).expand(t.size(0), -1)  # [batch_size, d_default]
    t_expanded = t.unsqueeze(1)  # [batch_size, 1]

    # 找到最近的索引
    nearest_index_sw = torch.argmin(torch.abs(midpoints_expanded - t_expanded), dim=1)  # [batch_size]

    # 贝叶斯更新
    final = sw_bayes(d_default, matrix_sw, nearest_index_sw, final)

    return final


def sw_transform(eps, d_default):
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    # report matrix
    m = d_default
    n = d_default
    m_cell = (1 + w) / m
    n_cell = 1 / n

    transform = np.ones((m, n)) * q * m_cell
    for i in range(n):
        left_most_v = (i * n_cell)
        right_most_v = ((i + 1) * n_cell)

        ll_bound = int(left_most_v / m_cell)
        lr_bound = int((left_most_v + w) / m_cell)
        rl_bound = int(right_most_v / m_cell)
        rr_bound = int((right_most_v + w) / m_cell)

        ll_v = left_most_v - w / 2
        rl_v = right_most_v - w / 2
        l_p = ((ll_bound + 1) * m_cell - w / 2 - ll_v) * (p - q) + q * m_cell
        r_p = ((rl_bound + 1) * m_cell - w / 2 - rl_v) * (p - q) + q * m_cell
        if rl_bound > ll_bound:
            transform[ll_bound, i] = (l_p - q * m_cell) * (
                    (ll_bound + 1) * m_cell - w / 2 - ll_v) / n_cell * 0.5 + q * m_cell
            transform[ll_bound + 1, i] = p * m_cell - (p * m_cell - r_p) * (
                    rl_v - ((ll_bound + 1) * m_cell - w / 2)) / n_cell * 0.5
        else:
            transform[ll_bound, i] = (l_p + r_p) / 2
            transform[ll_bound + 1, i] = p * m_cell

        lr_v = left_most_v + w / 2
        rr_v = right_most_v + w / 2
        r_p = (rr_v - (rr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        l_p = (lr_v - (lr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        if rr_bound > lr_bound:
            if rr_bound < m:
                transform[rr_bound, i] = (r_p - q * m_cell) * (
                        rr_v - (rr_bound * m_cell - w / 2)) / n_cell * 0.5 + q * m_cell

            transform[rr_bound - 1, i] = p * m_cell - (p * m_cell - l_p) * (
                    (rr_bound * m_cell - w / 2) - lr_v) / n_cell * 0.5

        else:
            transform[rr_bound, i] = (l_p + r_p) / 2
            transform[rr_bound - 1, i] = p * m_cell

        if rr_bound - 1 > ll_bound + 2:
            transform[ll_bound + 2: rr_bound - 1, i] = p * m_cell
    return transform
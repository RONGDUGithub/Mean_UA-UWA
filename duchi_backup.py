import numpy as np


def divide_interval_and_boundaries_sr(d_sr):
    # 检查d_sr是否大于0
    if d_sr <= 0:
        raise ValueError('d_sr must be a positive integer')

    # 生成分段的边界点
    edges = np.linspace(-1, 1, d_sr + 1)

    # 计算每段的中值
    mid_values = edges[:-1] + np.diff(edges) / 2

    # 创建返回矩阵BM，第一行是左边界点，第二行是右边界点
    BM = np.vstack((edges[:-1], edges[1:]))

    return mid_values, BM


def discretize_probability_sr(t, epsilon):
    # 定义分段函数的转折点
    p = (np.exp(epsilon) - 1) * t / (2 * np.exp(epsilon) + 2) + 1 / 2
    q = 1 - p

    # 初始化概率数组
    probabilities = np.array([p, q])

    return probabilities
import numpy as np
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import torch

def get_positions(a, b, A, n_segments=1024):
    """
    将区间 [a, b] 拆分成 n_segments 份，根据数组 A 中的值，返回位置参数（0 到 n_segments-1）。

    参数:
    a (float): 区间的起始点。
    b (float): 区间的结束点。
    A (array-like): 包含属于 [a, b] 区间的数值的数组。
    n_segments (int): 拆分的份数，默认值为 1024。

    返回:
    array: 包含位置参数的数组。
    """
    # 生成 n_segments + 1 个等间隔点
    edges = np.linspace(a, b, n_segments + 1)

    # 使用 np.digitize 确定 A 中每个值的位置参数
    positions = np.digitize(A, edges) - 1

    # 确保位置参数在合法范围内
    positions = np.clip(positions, 0, n_segments - 1)

    return positions


def elementwise_multiplication_three(vector1, vector2, vector3):
    # 检查向量长度是否相等
    if len(vector1) != len(vector2) or len(vector1) != len(vector3):
        raise ValueError("三个向量的长度必须相等")

    # 对应元素相乘
    result = np.multiply(np.multiply(vector1, vector2), vector3)

    return result


def kl_divergence(p, q):
    """
    Calculate the KL divergence between two distributions.

    Parameters:
    p (array-like): The first distribution (true distribution).
    q (array-like): The second distribution (approximate distribution).

    Returns:
    float: The KL divergence between distribution p and q.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Ensure the distributions sum to 1
    p /= np.sum(p)
    q /= np.sum(q)

    # Compute the KL divergence
    kl_div = np.sum(rel_entr(p, q))

    return kl_div


def js_divergence(p, q):
    """
    Calculate the Jensen-Shannon divergence between two distributions.

    Parameters:
    p (array-like): The first distribution.
    q (array-like): The second distribution.

    Returns:
    float: The JS divergence between distribution p and q.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Ensure the distributions sum to 1
    p /= np.sum(p)
    q /= np.sum(q)

    # Calculate the middle distribution
    m = 0.5 * (p + q)

    # Compute the JS divergence
    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    return js_div





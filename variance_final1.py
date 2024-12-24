import numpy as np
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import torch

# def Variance_sr(t, epsilon):
#     eexp = torch.exp(epsilon)
#     Var_sr = ((eexp + 1) / (eexp - 1)).pow(2) - t.pow(2)
#     return Var_sr


def Variance_sr(t, epsilon):
    # 把输入转成float64
    t = t.to(torch.float64)
    epsilon = epsilon.to(torch.float64)

    eexp = torch.exp(epsilon)
    Var_sr = ((eexp + 1) / (eexp - 1)).pow(2) - t.pow(2)

    # 确保输出是float64
    return Var_sr.to(torch.float64)

# def Variance_pm(t, epsilon):
#     eexp2 = torch.exp(epsilon / 2)
#     Var_pm = t.pow(2) / (eexp2 - 1) + (eexp2 + 3) / (3 * (eexp2 - 1).pow(2))
#     return Var_pm


def Variance_pm(t, epsilon):
    # 把输入转成float64
    t = t.to(torch.float64)
    epsilon = epsilon.to(torch.float64)

    eexp2 = torch.exp(epsilon / 2)
    Var_pm = t.pow(2) / (eexp2 - 1) + (eexp2 + 3) / (3 * (eexp2 - 1).pow(2))

    # 确保输出是float64
    return Var_pm.to(torch.float64)

# def Variance_laplace(t, epsilon):
#     # Compute a scalar variance
#     Var_laplace = 8 / epsilon.pow(2)
#     # Return a tensor of variances with the same shape as t
#     return Var_laplace * torch.ones_like(t)


def Variance_laplace(t, epsilon):
    # 把输入转成float64
    t = t.to(torch.float64)
    epsilon = epsilon.to(torch.float64)

    # Compute a scalar variance
    Var_laplace = 8 / epsilon.pow(2)

    # Return a tensor of variances with the same shape as t
    # ones_like会继承t的dtype(float64)
    return Var_laplace * torch.ones_like(t)

# def Variance_squarewave(t, epsilon):
#     t2 = (t + 1) / 2
#     eexp = torch.exp(epsilon)
#     budget = eexp
#     b = (epsilon * budget - budget + 1) / (2 * budget * (budget - 1 - epsilon))
#     p = budget / (2 * b * budget + 1)
#     q = 1 / (2 * b * budget + 1)
#     Var_sw = 4 * (q * ((1 + 3 * b + 3 * b.pow(2) - 6 * b * t2.pow(2)) / 3) +
#                   p * ((6 * b * t2.pow(2) + 2 * b.pow(3)) / 3) -
#                   ((t2 * 2 * b * (p - q) + q * (b + 1 / 2)).pow(2)))
#     Var_sw_unbiase = Var_sw / (2 * b * (p - q)).pow(2)
#     return Var_sw_unbiase


def Variance_squarewave(t, epsilon):
    # 把输入转成float64
    t = t.to(torch.float64)
    epsilon = epsilon.to(torch.float64)

    t2 = (t + 1) / 2
    eexp = torch.exp(epsilon)
    budget = eexp
    b = (epsilon * budget - budget + 1) / (2 * budget * (budget - 1 - epsilon))
    p = budget / (2 * b * budget + 1)
    q = 1 / (2 * b * budget + 1)

    Var_sw = 4 * (q * ((1 + 3 * b + 3 * b.pow(2) - 6 * b * t2.pow(2)) / 3) +
                  p * ((6 * b * t2.pow(2) + 2 * b.pow(3)) / 3) -
                  ((t2 * 2 * b * (p - q) + q * (b + 1 / 2)).pow(2)))

    Var_sw_unbiase = Var_sw / (2 * b * (p - q)).pow(2)

    # 确保输出是float64
    return Var_sw_unbiase.to(torch.float64)


def compute_pm_variance(final, epsilon, d_default, midpoints_minus1_to_1):
    t = midpoints_minus1_to_1[:d_default]
    variances = Variance_pm(t, epsilon)
    # var_final = torch.dot(final[:d_default], variances)
    var_final = torch.matmul(final[:, :d_default], variances)
    return var_final


def compute_sw_variance(final, epsilon, d_default, midpoints_minus1_to_1):
    t = midpoints_minus1_to_1[:d_default]
    variances = Variance_squarewave(t, epsilon)
    # var_final = torch.dot(final[:d_default], variances)
    var_final = torch.matmul(final[:, :d_default], variances)
    return var_final


def compute_laplace_variance(final, epsilon, d_default, midpoints_minus1_to_1):
    t = midpoints_minus1_to_1[:d_default]
    variances = Variance_laplace(t, epsilon)

    # Ensure both final[:d_default] and variances are 1D tensors
    # if final[:d_default].dim() != 1:
    #     raise ValueError("Expected final[:d_default] to be a 1D tensor")
    # if variances.dim() != 1:
    #     raise ValueError("Expected variances to be a 1D tensor")

    # var_final = torch.dot(final[:d_default], variances)
    var_final = torch.matmul(final[:, :d_default], variances)
    return var_final


def compute_duchi_variance(final, epsilon, d_default, midpoints_minus1_to_1):
    t = midpoints_minus1_to_1[:d_default]
    variances = Variance_sr(t, epsilon)
    # var_final = torch.dot(final[:, :d_default], variances)
    # variances = variances.reshape(-1, 1)  # 或者 .unsqueeze(1) VLDB new add
    print("final dtype:", final.dtype)
    print("variances dtype:", variances.dtype)
    var_final = torch.matmul(final[:, :d_default], variances)
    return var_final


def plot_final_vector(final, data0, i, batch_size):
    """
    绘制final向量的折线图，并在标题中显示data0的值。

    参数:
    - final: torch.Tensor，包含要绘制的最终向量
    - data0: torch.Tensor，包含额外的数据
    - i: int，索引值
    - batch_size: int，批量大小
    """
    final1 = final

    # 提取标量数据
    data1 = data0[(i - 1) * batch_size]
    data1 = data1.item()  # 转换为标量

    # 将 final 移动到 CPU 并转换为 numpy 数组
    final2 = final1.cpu().numpy()

    # 生成横坐标，[-1, 1] 等分 20 份
    x_coords = np.linspace(-1, 1, 20)
    print(x_coords)

    # 创建一个图形，调整图形大小为 (10, 6)
    plt.figure(figsize=(10, 6))

    # 确保 final2 是一维数组
    final2 = final2.squeeze()
    print(final2)

    # 绘制向量，确保是折线图
    plt.plot(x_coords, final2, marker='o', linestyle='-', color='b', label='final')

    # 添加标题和标签，标题中包含 data0 的值
    plt.title(f'Final Vector Visualization (data0: {data1:.4f})')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图像
    plt.show()
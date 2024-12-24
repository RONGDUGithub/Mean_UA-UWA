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

user_number = 2000
# 生成一个数据集：均匀分布（0,1000）数据量为1000000
# data = np.random.uniform(0, 1000, user_number)

# 生成符合 Beta(2, 5) 分布的数据
# data = np.random.beta(1, 6, user_number)

np.random.seed(0)

# 生成第一个峰值数据，均值为0，标准差为1，1000个样本
peak1 = np.random.normal(loc=0, scale=1, size=1000)

# 生成第二个峰值数据，均值为5，标准差为1，1000个样本
peak2 = np.random.normal(loc=5, scale=1, size=1000)

# 合并两组数据
data = np.concatenate([peak1, peak2])

# 如果需要将数据的范围调整到0到1000（假设你需要这样的范围）
# data_scaled = data * 1000

epsilon_default = 0.5

min_data = 0
max_data = 1
# NVP:输出的为扰动后的数组
# duchi:
result_duchi, noise_duchi = duchi(data, min_data, max_data, epsilon_default)
# pm
result_piecewise, noise_piecewise = piecewise(data, min_data, max_data, epsilon_default)
# laplace:
result_laplace, noise_laplace = laplace1(data, min_data, max_data, epsilon_default)
# sw_unbiased:
result_sw_unbiased, result_sw_biased = sw_unbiased(data, min_data, max_data, epsilon_default)  # 第一个变量：数据集    第二个变量：数据下界（这里是0） 第三个变量：数据上界（这里是1000） 第四个变量：隐私预算


d_default = 100
final = np.ones(d_default) / d_default
midpoints_minus1_to_1 = np.linspace(-1, 1 - 2 / d_default, d_default) + 1 / d_default
midpoints_minus0_to_1 = np.linspace(0, 1 - 1 / d_default, d_default) + 1 / (2 * d_default)

mean_final = [0] * user_number
final_aggre = [0] * d_default
for i in range(1, user_number + 1):
    # print(" i:", i)
    # duchi
    epsilon_duchi = epsilon_default
    # print("duchi_value:", noise_duchi[i - 1])
    final = duchi_bayes(noise_duchi[i - 1], midpoints_minus1_to_1, epsilon_duchi, final)

    # #laplace
    epsilon_lap = epsilon_default
    # print("lap_value:", noise_laplace[i - 1])
    final = laplace_bayes(d_default, epsilon_lap, noise_laplace[i - 1], final)

    # PM
    epsilon_pm = epsilon_default
    t = noise_piecewise[i - 1]
    # print("pm_value:", noise_piecewise[i - 1])
    final = pm_pre_bayes(t, epsilon_pm, d_default, final)

    # SW
    epsilon_sw = epsilon_default
    t = result_sw_biased[i - 1]
    # print("sw_value:", t*2-1)
    final = sw_pre_bayes(t, epsilon_sw, d_default, final)
    final_aggre += final
    print("final_aggre1:", final_aggre)

final_aggre = final_aggre / np.sum(final_aggre) if np.sum(final_aggre) != 0 else final_aggre
print("final_aggre2:", final_aggre)

normalized_value = 2 * (data[i - 1] - 0) / (1000 - 0) - 1
print("original_value:", normalized_value)
x = np.linspace(-1, 1, d_default)  # 生成从 0 到 1 的等间隔数值序列
plt.plot(x, final)
plt.show()

# # NVP:输出的为扰动后的分布（概率密度）此处使用的是EMS
distribution = sw(data, min_data, max_data, 3) / len(data)  # 第一个变量：数据集    第二个变量：数据下界（这里是0） 第三个变量：数据上界（这里是1000） 第四个变量：隐私预算
x = np.arange(0.5 * (max_data - min_data) / 1024, 1024.5 * (max_data - min_data) / 1024, (max_data - min_data) / 1024)
mean_sw_distribution = np.sum(np.multiply(distribution * len(data), x)) / len(data)

# #画图
x=np.linspace(0,1,1024)
plt.figure()
plt.plot(x,distribution)
# plt.plot(x2,final_aggre)
# plt.ylim(0,0.002)
plt.title('Distribution Estimation')
plt.show()

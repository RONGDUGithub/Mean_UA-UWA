import numpy as np
import matplotlib.pyplot as plt
import openpyxl as openpyxl
import xlrd as xlrd

from Duchinvp import duchi
from Hybridnvp import hybrid
from Laplacenvp import laplace
from Piecewisenvp import piecewise
from SWunbiasednvp import sw_unbiased
from sw import sw
from entropyfile import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 确保安装了 openpyxl 或者 xlrd 库
# pip install openpyxl
# pip install xlrd

import pandas as pd

# 读取Excel文件
df = pd.read_excel('taxi.xlsx')

# 获取第一列数据并转换为数组
data = df.iloc[:, 0].to_numpy()
min_data = min(data)
max_data = max(data)
#生成一个数据集：均匀分布（0,1000）数据量为1000000

# data=np.random.uniform(0,1000,1000000)
privacy_budget = 1

privacy_budget2 = 0.1

#NVP:输出的为扰动后的数组
#duchi:
result_duchi=duchi(data,min_data,max_data,0.1)#第一个变量：数据集    第二个变量：数据下界（这里是0） 第三个变量：数据上界（这里是1000） 第四个变量：隐私预算
#piecewise:
result_piecewise=duchi(data,min_data,max_data,0.5)#第一个变量：数据集    第二个变量：数据下界（这里是0） 第三个变量：数据上界（这里是1000） 第四个变量：隐私预算
#hybrid:
result_hybrid=duchi(data,min_data,max_data,1)#第一个变量：数据集    第二个变量：数据下界（这里是0） 第三个变量：数据上界（这里是1000） 第四个变量：隐私预算
#laplace:
result_laplace=duchi(data,min_data,max_data,2)#第一个变量：数据集    第二个变量：数据下界（这里是0） 第三个变量：数据上界（这里是1000） 第四个变量：隐私预算
#sw_unbiased:
result_sw_unbiased=duchi(data,min_data,max_data,5)#第一个变量：数据集    第二个变量：数据下界（这里是0） 第三个变量：数据上界（这里是1000） 第四个变量：隐私预算

# SW 没有跑呀

mean_groundtruth = np.mean(data, axis=0)
mean_duchi = np.mean(result_duchi, axis=0)
mean_piecewise = np.mean(result_piecewise, axis=0)
mean_hybrid = np.mean(result_hybrid, axis=0)
mean_laplace = np.mean(result_laplace, axis=0)
mean_sw_unbiased = np.mean(result_sw_unbiased, axis=0)

print("均值：", mean_groundtruth, mean_duchi, mean_piecewise, mean_hybrid, mean_laplace, mean_sw_unbiased)

mean_vector = [mean_duchi, mean_piecewise, mean_hybrid, mean_laplace, mean_sw_unbiased]

# 将向量集合存储在一个列表中
vectors = [result_duchi, result_piecewise, result_hybrid, result_laplace, result_sw_unbiased]

entropy_duchi = calculate_entropy(result_duchi)
entropy_piecewise = calculate_entropy(result_piecewise)
entropy_hybrid = calculate_entropy(result_hybrid)
entropy_laplace = calculate_entropy(result_laplace)
entropy_sw_unbiased = calculate_entropy(result_sw_unbiased)


print("熵", entropy_duchi, entropy_piecewise, entropy_hybrid, entropy_laplace, entropy_sw_unbiased)

entropy = [entropy_duchi, entropy_piecewise, entropy_hybrid, entropy_laplace, entropy_sw_unbiased]
entropy = np.array(entropy)
# 检查是否存在无效值（NaN）

if np.isnan(entropy).any():
    print("存在无效值（NaN）")
else:
    # 计算权重
    weights = calculate_weights(entropy)

    # 计算加权均值
    # weighted_mean = calculate_weighted_mean(vectors, weights)
    # 计算加权均值
    weighted_average = np.average(mean_vector, weights=weights)

    print("加权权重：", weights)
    print("加权均值：", weighted_average)
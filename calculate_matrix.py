import numpy as np


def calculate_matrix_sw(epsilon_sw, db_sw, d_sw, midpoints_minus0_to_1_sw, BM_sw):
    b_sw = ((epsilon_sw * np.exp(epsilon_sw) - np.exp(epsilon_sw) + 1) /
            (2 * np.exp(epsilon_sw) * (np.exp(epsilon_sw) - 1 - epsilon_sw)))
    p_sw = np.exp(epsilon_sw) / (2 * b_sw * np.exp(epsilon_sw) + 1)
    q_sw = 1 / (2 * b_sw * np.exp(epsilon_sw) + 1)
    Matrix_sw = np.zeros((db_sw, len(midpoints_minus0_to_1_sw)))

    for i, t in enumerate(midpoints_minus0_to_1_sw):
        # Calculate probabilities for each segment based on t
        probabilities = np.zeros(db_sw)
        for j in range(db_sw):
            probabilities[j] = p_sw if BM_sw[j] < t else q_sw
        Matrix_sw[:, i] = probabilities

    return Matrix_sw


# def generate_probability_matrix_single_function(epsilon_pm, db_pm, d_pm, midpoints_minus1_to_1_pm, BM_pm):
#     C = (np.exp(epsilon_pm / 2) + 1) / (np.exp(epsilon_pm / 2) - 1)
#     p_pm = (np.exp(epsilon_pm / 2) / (np.exp(epsilon_pm / 2) + 1)) / (C - 1)
#     q_pm = (1 - (np.exp(epsilon_pm / 2) / (np.exp(epsilon_pm / 2) + 1))) / (1 + C)
#
#     matrix_pm = np.zeros((db_pm, d_pm))
#
#     for i, t in enumerate(midpoints_minus1_to_1_pm):
#         for j in range(db_pm):
#             matrix_pm[j, i] = p_pm if BM_pm[j] < t else q_pm
#
#     return matrix_pm
# import numpy as np

def discretize_probability_pm(C, t, p, q, db_pm, BM):
    # 定义分段函数的转折点
    breakpoint1 = (C + 1) * t / 2 - (C - 1) / 2
    breakpoint2 = breakpoint1 + (C - 1)

    # 初始化概率数组
    probabilities = np.zeros(db_pm)

    # 遍历每个段，基于概率密度函数计算概率
    for i in range(db_pm):
        # 获取段的起始和结束点
        segment_start = BM[i, 0]
        segment_end = BM[i, 1]

        # 初始化段概率为0
        segment_prob = 0

        # 当段在第一段区间内
        if segment_start < breakpoint1 and segment_end <= breakpoint1:
            segment_prob = q * (segment_end - segment_start)

        # 当段完全在第二段区间内
        elif segment_start >= breakpoint1 and segment_end <= breakpoint2:
            segment_prob = p * (segment_end - segment_start)

        # 当段完全在第三段区间内
        elif segment_start >= breakpoint2 and segment_end > breakpoint2:
            segment_prob = q * (segment_end - segment_start)

        # 当段跨越第一和第二段区间
        elif segment_start < breakpoint1 and segment_end > breakpoint1 and segment_end <= breakpoint2:
            segment_prob = q * (breakpoint1 - segment_start) + p * (segment_end - breakpoint1)

        # 当段跨越第二和第三段区间
        elif segment_start >= breakpoint1 and segment_start < breakpoint2 and segment_end > breakpoint2:
            segment_prob = p * (breakpoint2 - segment_start) + q * (segment_end - breakpoint2)

        # 当段跨越所有三个区间
        elif segment_start < breakpoint1 and segment_end > breakpoint2:
            segment_prob = q * (breakpoint1 - segment_start) + p * (breakpoint2 - breakpoint1) + q * (segment_end - breakpoint2)

        # 将计算出的段概率赋值
        probabilities[i] = segment_prob

    # 归一化概率使其总和为1
    probabilities = probabilities / np.sum(probabilities)

    return probabilities

def compute_matrix_pm(epsilon_pm, midpoints_minus1_to_1_pm, db_pm, BM_pm):
    C = (np.exp(epsilon_pm / 2) + 1) / (np.exp(epsilon_pm / 2) - 1)
    p_pm = (np.exp(epsilon_pm / 2) / (np.exp(epsilon_pm / 2) + 1)) / (C - 1)
    q_pm = (1 - (np.exp(epsilon_pm / 2) / (np.exp(epsilon_pm / 2) + 1))) / (1 + C)
    Matrix_pm = np.zeros((db_pm, len(midpoints_minus1_to_1_pm)))

    for i, t in enumerate(midpoints_minus1_to_1_pm):
        probabilities = discretize_probability_pm(C, t, p_pm, q_pm, db_pm, BM_pm)
        Matrix_pm[:, i] = probabilities

    return Matrix_pm


def compute_matrix_sw(epsilon_sw, midpoints_minus0_to_1_sw, db_sw, BM_sw):
    b_sw = (((epsilon_sw * np.exp(epsilon_sw) - np.exp(epsilon_sw) + 1)) / (2 * np.exp(epsilon_sw) * (np.exp(epsilon_sw) - 1 - epsilon_sw)))
    p_sw = np.exp(epsilon_sw) / (2 * b_sw * np.exp(epsilon_sw) + 1)
    q_sw = 1 / (2 * b_sw * np.exp(epsilon_sw) + 1)

    Matrix_sw = np.zeros((db_sw, len(midpoints_minus0_to_1_sw)))

    for i in range(len(midpoints_minus0_to_1_sw)):
        t = midpoints_minus0_to_1_sw[i]
        probabilities = discretize_probability_sw(b_sw, t, p_sw, q_sw, db_sw, BM_sw)
        Matrix_sw[:, i] = probabilities

    return Matrix_sw

def discretize_probability_sw(b_sw, t, p, q, db_sw, BM):
    breakpoint1 = t - b_sw
    breakpoint2 = t + b_sw

    probabilities = np.zeros(db_sw)

    for i in range(db_sw):
        segment_start = BM[i, 0]
        segment_end = BM[i, 1]

        segment_prob = 0

        if segment_start < breakpoint1 and segment_end <= breakpoint1:
            segment_prob = q * (segment_end - segment_start)

        elif segment_start >= breakpoint1 and segment_end <= breakpoint2:
            segment_prob = p * (segment_end - segment_start)

        elif segment_start >= breakpoint2 and segment_end > breakpoint2:
            segment_prob = q * (segment_end - segment_start)

        elif segment_start < breakpoint1 and segment_end > breakpoint1 and segment_end <= breakpoint2:
            segment_prob = q * (breakpoint1 - segment_start) + p * (segment_end - breakpoint1)

        elif segment_start >= breakpoint1 and segment_start < breakpoint2 and segment_end > breakpoint2:
            segment_prob = p * (breakpoint2 - segment_start) + q * (segment_end - breakpoint2)

        elif segment_start < breakpoint1 and segment_end > breakpoint2:
            segment_prob = q * (breakpoint1 - segment_start) + p * (breakpoint2 - breakpoint1) + q * (segment_end - breakpoint2)

        probabilities[i] = segment_prob

    probabilities = probabilities / np.sum(probabilities)
    return probabilities
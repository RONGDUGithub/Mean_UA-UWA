import numpy as np


def discretize_interval_pm(C, d_pm):
    # 计算每段长度 o
    o = 2*C / (d_pm)
    h = 2 / d_pm
    # # 计算 h 的值
    # h = np.ceil((C - 1) / o)
    #
    # # 判断是否 h 等于 (C-1)/o
    # if h == (C - 1) / o:
    #     # h 恰好等于 (C-1)/o 时
    #     # [-C, -1] 和 [1, C] 区间的每段长度都是 o
    #     left_segment_length = o
    #     right_segment_length = o
    # else:
    #     # h 不等于 (C-1)/o 时
    #     # [-C, -1] 区间的第一段长度和 [1, C] 区间的最后一段长度不是 o
    #     left_segment_length = (C - 1) - (h - 1) * o
    #     right_segment_length = (C - 1) - (h - 1) * o

    # 初始化边界矩阵 BM 和中点数组 midpoints
    db_pm = d_pm   # 总段数
    BM = np.zeros((db_pm, 2))  # 每行存储一个段的左边界和右边界
    midpoints = np.zeros(db_pm)  # 存储每个段的中点

    # 定义辅助函数来填充边界矩阵和中点数组
    def fill_segments(start_index, start_value, num_segments, segment_length):
        for i in range(num_segments):
            left = start_value + i * segment_length
            right = left + segment_length
            # print('i', i)
            # print('start_index', start_index)
            BM[start_index + i, :] = [left, right]
            midpoints[start_index + i] = (left + right) / 2

    # 填充 [-C, -1] 区间
    # fill_segments(0, -C, 1, left_segment_length)
    # fill_segments(1, -C + left_segment_length, int(h - 1), o)

    # 填充 [-1, 1] 区间
    fill_segments(0, -C, 1024, o)

    def get_midpoints(n_segments, start=-1, end=1):
        # 生成 n_segments + 1 个等间隔点
        edges = np.linspace(start, end, n_segments + 1)

        # 计算相邻区间的中值
        midpoints = (edges[:-1] + edges[1:]) / 2

        return midpoints

    # 将 [-1, 1] 区间均分成 1024 份
    n_segments = 1024
    midpoints = get_midpoints(n_segments)


    midpoints_minus1_to_1 = midpoints

    return midpoints, BM, midpoints_minus1_to_1


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
            segment_prob = q * (breakpoint1 - segment_start) + p * (breakpoint2 - breakpoint1) + q * (
                        segment_end - breakpoint2)

        # 将计算出的段概率赋值
        probabilities[i] = segment_prob

    # 归一化概率使其总和为1
    probabilities /= np.sum(probabilities)

    return probabilities

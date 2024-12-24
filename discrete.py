import numpy as np
import math


def divide_interval_and_boundaries_sr(d_sr):
    """
    Divides the interval [-1, 1] into d_sr segments and calculates their mid-values and boundaries.

    Args:
    d_sr: number of segments (must be a positive integer)

    Returns:
    mid_values: array of mid-values for each segment
    BM: 2-row array where the first row is the start and the second row is the end of each segment
    """
    # Check if d_sr is a positive integer
    if d_sr <= 0:
        raise ValueError('d_sr must be a positive integer')

    # Generate the boundaries of the segments
    edges = np.linspace(-1, 1, d_sr + 1)

    # Calculate the mid-values of each segment
    mid_values = edges[:-1] + np.diff(edges) / 2

    # Create the BM matrix with start and end points of each segment
    BM = np.array([edges[:-1], edges[1:]])

    return mid_values, BM


import numpy as np


def discretize_interval_pm(C, d_pm):
    """
    Discretizes the interval [-C, C] into segments of varying lengths based on the parameters.

    Args:
    C: the half-range of the interval
    d_pm: number of segments in the central interval [-1, 1]

    Returns:
    midpoints: midpoints of all segments
    BM: boundary matrix of all segments
    midpoints_minus1_to_1: midpoints of the segments within [-1, 1]
    db_pm: total number of segments
    """
    # Calculate segment length o
    o = 2*C / (d_pm)

    # Calculate h value
    # h = int(np.ceil((C - 1) / o))



    # Initialize boundary matrix BM and midpoints array
    db_pm = d_pm
    BM = np.zeros((db_pm, 2))
    midpoints = np.zeros(db_pm)

    # Function to fill segments in BM and midpoints
    def fill_segments(start_index, start_value, num_segments, segment_length):
        for i in range(num_segments):
            left = start_value + i * segment_length
            right = left + segment_length
            BM[start_index + i, :] = [left, right]
            midpoints[start_index + i] = (left + right) / 2

    # Fill [-C, -1] interval
    # fill_segments(0, -C, 1, left_segment_length)
    # fill_segments(1, -C + left_segment_length, h - 1, o)

    # Fill [-1, 1] interval
    fill_segments(0, -C, d_pm, o)

    # # Fill [1, C] interval
    # fill_segments(h + d_pm, 1, h - 1, o)
    # fill_segments(h + d_pm + h - 1, C - right_segment_length, 1, right_segment_length)

    # Extract midpoints of the [-1, 1] interval
    points = np.linspace(-1, 1, d_pm + 1)

    # 计算相邻点的中点
    midpoints_minus1_to_1 = (points[:-1] + points[1:]) / 2

    return midpoints, BM, midpoints_minus1_to_1, db_pm



def discretize_interval_sw(b, d_sw):
    """
    Discretizes the intervals [-b, 0] and [0, 1+b] into segments.

    Args:
    b (float): The boundary value for the intervals.
    d_sw (int): Number of segments in the [0, 1] interval.

    Returns:
    tuple: A tuple containing:
        - midpoints (numpy.ndarray): An array of midpoints for all segments.
        - BM (numpy.ndarray): A matrix where each row contains the start and end of each segment.
        - midpoints_minus0_to_1 (numpy.ndarray): An array of midpoints for the [0, 1] interval.
        - db_sw (int): Total number of segments.
    """
    # Calculate segment length o
    o = (1+2*b) / d_sw
    #
    # # Calculate h value
    # h = math.ceil(b / o)
    #
    # # Set segment lengths for the first and last segments if h is not exactly b/o
    # if h == (b) / o:
    #     left_segment_length = o
    #     right_segment_length = o
    # else:
    #     left_segment_length = b - (h - 1) * o
    #     right_segment_length = b - (h - 1) * o

    # Initialize boundary matrix BM and midpoints array
    db_sw =  d_sw  # Total number of segments
    BM = np.zeros((db_sw, 2))
    midpoints = np.zeros(db_sw)

    # Helper function to fill segments in BM and midpoints
    def fill_segments(start_index, start_value, num_segments, segment_length):
        for i in range(num_segments):
            left = start_value + i * segment_length
            right = left + segment_length
            BM[start_index + i, :] = [left, right]
            midpoints[start_index + i] = (left + right) / 2

    # # Fill [-b, 0] interval
    # fill_segments(0, -b, 1, left_segment_length)
    # fill_segments(1, -b + left_segment_length, h - 1, o)

    # Fill [0, 1] interval
    fill_segments(0, -b, d_sw, o)

    # Fill [1, 1+b] interval
    # fill_segments(h + d_sw, 1, h - 1, o)
    # fill_segments(h + d_sw + h - 1, 1 + b - right_segment_length, 1, right_segment_length)

    # Extract midpoints of the [0, 1] interval
    # midpoints_minus0_to_1 = midpoints[h:h + d_sw]
    points = np.linspace(-b, 1+b, d_sw + 1)

    # 计算相邻点的中点
    midpoints_minus0_to_1 = (points[:-1] + points[1:]) / 2

    return midpoints, BM, midpoints_minus0_to_1, db_sw



import numpy as np
import random
import scipy
from numpy import linalg as LA


def linear_normalize(array,min_val,max_val):
    normalized_array = 2 * (array - min_val) / (max_val - min_val) - 1
    return normalized_array


def linear_denormalize(normalized_array,min_val,max_val):
    denormalized_array = 0.5 * (normalized_array + 1) * (max_val - min_val) + min_val
    return denormalized_array


def disturb_data(data,ee):
    lv = (ee*data-1)/(ee-1)
    rv = (ee*data+1)/(ee-1)
    s=(ee+1)/(ee-1)

    if random.random() < ee/(ee+1):
        return random.uniform(lv, rv)
    else:
        return disturb_O(data,rv,lv,s)


def disturb_O(data,rv,lv,s):
    if random.random() < (data+1)/2:
        data=random.uniform(-s, lv)
    else:
        data=random.uniform(rv, s)
    return data


def piecewise(inputdata,min_val,max_val,epsilon):
    inputdata[inputdata > max_val] = max_val
    inputdata[inputdata < min_val] = min_val
    normalized_data=linear_normalize(inputdata,min_val,max_val)
    ee = np.exp(epsilon/2)
    for i in range(len(normalized_data)):
        normalized_data[i]=disturb_data(normalized_data[i],ee)
    # max_value = np.max(normalized_data)

    # print("The maximum value in the array is:", max_value)

    # min_value = np.min(normalized_data)

    # print("The maximum value in the array is:", min_value)
    return linear_denormalize(normalized_data, min_val, max_val), normalized_data


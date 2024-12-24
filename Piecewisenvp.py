import torch
import random

def linear_normalize(array, min_val, max_val):
    normalized_array = 2 * (array - min_val) / (max_val - min_val) - 1
    return normalized_array

def linear_denormalize(normalized_array, min_val, max_val):
    denormalized_array = 0.5 * (normalized_array + 1) * (max_val - min_val) + min_val
    return denormalized_array

def disturb_data(data, ee):
    lv = (ee * data - 1) / (ee - 1)
    rv = (ee * data + 1) / (ee - 1)
    s = (ee + 1) / (ee - 1)

    if random.random() < ee / (ee + 1):
        return torch.tensor([random.uniform(lv, rv)], device='cuda', dtype=torch.float32).item()
    else:
        return disturb_O(data, rv, lv, s)

def disturb_O(data, rv, lv, s):
    if random.random() < (data + 1) / 2:
        data = torch.tensor([random.uniform(-s, lv)], device='cuda', dtype=torch.float32).item()
    else:
        data = torch.tensor([random.uniform(rv, s)], device='cuda', dtype=torch.float32).item()
    return data

def piecewise(inputdata, min_val, max_val, epsilon):
    inputdata = torch.clamp(inputdata, min_val, max_val)  # Clamp data to range
    normalized_data = linear_normalize(inputdata, min_val, max_val)
    ee = torch.exp(epsilon.clone().detach() / 2)

    disturbed_data = torch.zeros_like(normalized_data)
    for i in range(len(normalized_data)):
        disturbed_data[i] = disturb_data(normalized_data[i], ee)

    return linear_denormalize(disturbed_data, min_val, max_val), disturbed_data

print("new round:")

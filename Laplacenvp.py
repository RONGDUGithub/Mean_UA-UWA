import torch

def laplace_noisy(epsilon, device='cuda'):
    # 使用 PyTorch 的随机数生成器来生成拉普拉斯噪声
    n_value = torch.distributions.laplace.Laplace(0, 2 / epsilon).sample().to(device)
    return n_value

def laplace_mech(data, epsilon):
    for i in range(len(data)):
        data[i] += laplace_noisy(epsilon)
    return data

def linear_normalize(array, min_val, max_val):
    normalized_array = (array - min_val) * 2 / (max_val - min_val) - 1
    return normalized_array

def linear_denormalize(normalized_array, min_val, max_val):
    denormalized_array = 0.5 * (normalized_array + 1) * (max_val - min_val) + min_val
    return denormalized_array

def laplace1(data, min_val, max_val, epsilon):
    data = torch.clamp(data, min_val, max_val)  # 限制数据范围
    normalized_data = linear_normalize(data, min_val, max_val)
    data_noisy = laplace_mech(normalized_data, epsilon)
    data_noisy_denor = linear_denormalize(data_noisy, min_val, max_val)
    return data_noisy_denor, data_noisy


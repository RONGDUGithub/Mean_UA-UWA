import torch


def sw_unbiased(ori_samples, l, h, eps):
    # Clone the original samples to avoid modifying the input tensor directly
    ori_samples = ori_samples.clone()

    # Clamp the values to the specified range [l, h]
    ori_samples = torch.clamp(ori_samples, l, h)

    # Compute constants using the provided epsilon
    ee = torch.exp(eps.clone().detach())
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    # Normalize the samples to range [0, 1]
    samples = (ori_samples - l) / (h - l)

    # Generate random numbers for each sample
    randoms = torch.rand_like(samples)

    # Initialize tensor for noisy samples
    noisy_samples = torch.zeros_like(samples)

    # Compute noisy samples based on conditions
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2

    index = (randoms > (q * samples)) & (randoms <= (q * samples + p * w))
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2

    index = randoms > (q * samples + p * w)
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2

    # Compute the final noisy samples
    b = w / 2
    k = q * (b + 0.5)
    c = noisy_samples - k
    final = c / (2 * b * (p - q))
    final = final * (h - l) + l

    return final, noisy_samples


def sw_biased(ori_samples, l, h, eps):
    """
    实现有偏滑动窗口机制，使用 PyTorch 在 GPU 上运行。

    参数:
    - ori_samples: 原始样本数据 (PyTorch Tensor)
    - l: 样本数据的最小值 (标量)
    - h: 样本数据的最大值 (标量)
    - eps: 隐私参数 epsilon (标量)

    返回:
    - noisy_samples: 添加噪声后的样本数据 (PyTorch Tensor)
    """

    ee = torch.exp(torch.tensor(eps, device=ori_samples.device))
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    samples = (ori_samples - l) / (h - l)
    randoms = torch.rand_like(samples)

    noisy_samples = torch.zeros_like(samples)

    # 计算噪声样本数据
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2

    return noisy_samples



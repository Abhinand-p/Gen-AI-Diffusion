import torch


def add_noise(imgs):
    """
    Adds random Gaussian noise to the input images.

    Parameters:
        imgs (torch.Tensor): A batch of images in tensor format.

    Returns:
        torch.Tensor: Noisy images with a blend of original and Gaussian noise.
    """
    dev = imgs.device
    percent = 0.5
    beta = torch.tensor(percent, device=dev)  # Weight for noise contribution
    alpha = torch.tensor(1 - percent, device=dev)  # Weight for original image contribution
    noise = torch.randn_like(imgs)  # Generate random noise with the same shape as imgs
    return alpha * imgs + beta * noise  # Blend the original image with the noise
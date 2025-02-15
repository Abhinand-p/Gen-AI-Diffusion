import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import torch
import os


def show_save_images(dataset, num_samples=10, save_path="Visualisation", save_filename="samples.png"):
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))  # Create a figure to hold all images

    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        axes[i].imshow(torch.squeeze(img[0]), cmap='gray')
        axes[i].axis('off')

    plt.tight_layout()
    save_full_path = os.path.join(save_path, save_filename)
    plt.savefig(save_full_path, bbox_inches='tight', dpi=300)
    print(f"Saved all {num_samples} images in one file: {save_full_path}")
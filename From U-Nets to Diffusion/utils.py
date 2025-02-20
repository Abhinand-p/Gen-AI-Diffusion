import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import torch
import os
import torchvision.transforms as transforms
import graphviz

from torchview import draw_graph


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


def visualize_model_graph(model, batch_size, img_channels, img_size, device, scale=1.5,
                          save_path='Visualisation/model_graph.png'):
    graphviz.set_jupyter_format('png')
    model_graph = draw_graph(
        model,
        input_size=(batch_size, img_channels, img_size, img_size),
        device=device,
        expand_nested=True
    )
    model_graph.resize_graph(scale=scale)
    model_graph.visual_graph.render(save_path.replace('.png', ''), format='png', cleanup=True)
    return model_graph.visual_graph


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])
    plt.ion()
    plt.imshow(reverse_transforms(image[0].detach().cpu()))

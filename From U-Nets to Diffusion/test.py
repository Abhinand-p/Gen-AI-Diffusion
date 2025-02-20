import os
import torch
import matplotlib.pyplot as plt

from utils import show_tensor_image


@torch.no_grad()  # Disable gradient computation for inference
def test(model, img_channels, img_size, device, save_path="Visualisation/Generated_images.png", num_samples=10):
    """
    Generates and visualizes `num_samples` images using a trained model.

    Parameters:
        model (torch.nn.Module): Trained generative model.
        img_channels (int): Number of image channels (e.g., 1 for grayscale, 3 for RGB).
        img_size (int): Height and width of the generated images.
        device (torch.device): Device to run the model on (CPU or GPU).
        save_path (str, optional): Path to save the generated images grid. Defaults to "Visualisation/Generated_images.png".
        num_samples (int, optional): Number of images to generate. Defaults to 10.
    """

    os.makedirs("Visualisation", exist_ok=True)

    noise_samples = []
    generated_samples = []

    for _ in range(num_samples):
        # Create a random noise tensor with the specified shape and send it to the device
        noise = torch.randn((1, img_channels, img_size, img_size), device=device)

        # Pass the noise through the model to generate an image
        result = model(noise)

        noise_samples.append(noise)
        generated_samples.append(result)

    # Create a new figure with the defined size
    nrows = num_samples
    ncols = 2
    figsize = (6, 2 * num_samples)
    plt.figure(figsize=figsize)

    for i in range(num_samples):
        samples = {
            "Noise": noise_samples[i],
            "Generated Image": generated_samples[i]
        }

        for j, (title, img) in enumerate(samples.items()):
            ax = plt.subplot(nrows, ncols, i * ncols + j + 1)
            ax.set_title(title)
            ax.axis("off")
            show_tensor_image(img)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Generated images saved to {save_path}")
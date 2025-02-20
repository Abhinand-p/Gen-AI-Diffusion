import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.optim import Adam
from unet import UNet
from dataset import get_dataloader
from utils import show_tensor_image, visualize_model_graph
from noise import add_noise

# Defining image and batch parameters
IMG_SIZE = 16
IMG_CH = 1
BATCH_SIZE = 128

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# Adjust PyTorch settings for performance
torch._dynamo.config.cache_size_limit = 64  # Increase cache limit to avoid warnings
torch.backends.cudnn.allow_tf32 = True  # Enable TensorFloat-32 computations for better performance


def get_loss(model, imgs):
    """
    Computes the Mean Squared Error (MSE) loss between the original and denoised images.

    Args:
        model: The UNet model used for denoising.
        imgs: Batch of original images.

    Returns:
        Computed MSE loss.
    """
    imgs_noisy = add_noise(imgs)  # Add noise to images
    imgs_pred = model(imgs_noisy)  # Predict denoised images
    return F.mse_loss(imgs, imgs_pred)  # Compute MSE loss


@torch.no_grad()
def plot_sample(model, imgs, epoch, save_path="Visualisation/Training_progress.png", max_epochs=5):
    """
    Plots and saves the original, noisy, and denoised images for visualization.

    Args:
        model: The trained UNet model.
        imgs: A batch of images to visualize.
        epoch: Current training epoch.
        save_path: Path to save the visualization.
        max_epochs: Total number of epochs for training.
    """
    os.makedirs("Visualisation", exist_ok=True)

    imgs = imgs[[0], :, :, :]  # Select the first image from the batch
    imgs_noisy = add_noise(imgs[[0], :, :, :])
    imgs_pred = model(imgs_noisy)

    # Define number of rows and columns for the plot
    nrows = max_epochs
    ncols = 3
    figsize = (10, 2 * max_epochs)

    if epoch == 0:
        plt.figure(figsize=figsize)

    # Dictionary mapping image types to their tensors
    samples = {
        "Original": imgs,
        "Noise Added": imgs_noisy,
        "Predicted Original": imgs_pred
    }

    # Plot images
    for i, (title, img) in enumerate(samples.items()):
        ax = plt.subplot(nrows, ncols, epoch * ncols + i + 1)
        ax.set_title(f"{title} (Epoch {epoch})")
        ax.axis("off")
        show_tensor_image(img)

    # Save the plot after the last epoch
    if epoch == max_epochs - 1:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def train(model, dataloader, max_epochs=5):
    """
    Trains the UNet model using Adam optimizer and MSE loss.

    Args:
        model: The UNet model to train.
        dataloader: DataLoader providing training images.
        max_epochs: Number of training epochs.
    """
    optimizer = Adam(model.parameters(), lr=0.0001)  # Adam optimizer with learning rate 0.0001

    model.train()  # Set model to training mode
    for epoch in range(max_epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()  # Zero the gradients
            images = batch[0].to(device)  # Move images to the selected device
            loss = get_loss(model, images)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            if step % 100 == 0:
                print(f"Epoch {epoch} | Step {step:03d} Loss: {loss.item()}")

        plot_sample(model, images, epoch, max_epochs=max_epochs)


if __name__ == "__main__":
    dataloader = get_dataloader()
    model = torch.compile(UNet().to(device))
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    visualize_model_graph(model, BATCH_SIZE, IMG_CH, IMG_SIZE, device)
    train(model, dataloader)
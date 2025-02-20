import torch

from unet import UNet
from dataset import get_dataloader
from utils import visualize_model_graph
from train import train
from test import test

# Defining image and batch parameters
IMG_SIZE = 16
IMG_CH = 1
BATCH_SIZE = 128
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataloader = get_dataloader()
    model = torch.compile(UNet().to(device))
    visualize_model_graph(model, BATCH_SIZE, IMG_CH, IMG_SIZE, device)

    print("Training model...")
    train(model, dataloader)

    print("Testing model...")
    model.eval()
    test(model, IMG_CH, IMG_SIZE, device)

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from utils import show_save_images

IMG_SIZE = 16
IMG_CH = 1
BATCH_SIZE = 128

def load_fashionMNIST(data_transform, train=True):
    """
    Loads the FashionMNIST dataset with the specified transformations.

    Parameters:
        data_transform (torchvision.transforms.Compose): Transformation pipeline for preprocessing images.
        train (bool): If True, loads the training dataset; otherwise, loads the test dataset.

    Returns:
        torchvision.datasets.FashionMNIST: The dataset object.
    """
    return torchvision.datasets.FashionMNIST(
        "./data",
        download=True,  # Download the dataset if not already available
        train=train,
        transform=data_transform,
    )
def load_transformed_fashionMNIST():
    """
    Applies transformations to the FashionMNIST dataset and loads both the training and test sets.

    Returns:
        torch.utils.data.ConcatDataset: Combined dataset (training + test) with applied transformations.
    """
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize images to 16x16
        transforms.ToTensor(),  # Scales data between 0 and 1.
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally (data augmentation).
        # Retail from vertical flip as we do not want to generate inverted images
        transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize pixel values between -1 and 1.
    ]
    data_transform = transforms.Compose(data_transforms)  # Combine transformations into a pipeline
    train_set = load_fashionMNIST(data_transform, train=True)
    show_save_images(train_set, 10, "Visualisation", "Training_samples.png")
    test_set = load_fashionMNIST(data_transform, train=False)
    show_save_images(test_set, 10, "Visualisation", "Testing_samples.png")
    return torch.utils.data.ConcatDataset([train_set, test_set])  # Merge both datasets into a single dataset

def get_dataloader():
    """
    Loads the transformed FashionMNIST dataset and creates a DataLoader for batch processing.

    Returns:
        DataLoader: DataLoader object for iterating over the dataset in batches.
    """
    data = load_transformed_fashionMNIST()
    dataloader = DataLoader(data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,  # Shuffle data at each epoch for randomness
                            drop_last=True  # Drop the last batch if it contains fewer samples than batch_size
                            )
    return dataloader


if __name__ == "__main__":
    loader = get_dataloader()

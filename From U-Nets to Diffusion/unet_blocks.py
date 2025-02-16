import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """
    A downsampling block for a U-Net architecture.
    This block applies two convolutional layers, followed by ReLU activations and batch normalization.
    Finally, it downsamples the feature maps using MaxPooling.
    """

    def __init__(self, in_ch, out_ch):
        """
        Initializes the DownBlock.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """

        # Define convolution parameters
        kernel_size = 3  # Standard 3x3 convolution kernel size
        stride = 1  # Stride of 1 to preserve spatial dimensions
        padding = 1  # Padding of 1 to maintain the same feature map size after convolution

        super().__init__()

        # Defining the sequence of layers for the downsampling block
        layers = [
            # First Convolutional layer: extracts features while keeping the spatial size unchanged
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),  # Batch normalization for stability and faster convergence
            nn.ReLU(),  # ReLU activation to introduce non-linearity

            # Second Convolutional layer: further refines extracted features
            nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),

            # MaxPooling layer: Reduces spatial dimensions by a factor of 2
            nn.MaxPool2d(kernel_size=2)  # Reduces width and height by half
        ]

        # Store the layers as a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the DownBlock.

        Args:
            x (Tensor): Input feature map of shape (batch_size, in_ch, height, width)

        Returns:
            Tensor: Downsampled feature map of shape (batch_size, out_ch, height/2, width/2)
        """
        return self.model(x)  # Pass input through the sequential layers


class UpBlock(nn.Module):
    """
    An upsampling block for the U-Net architecture.
    This block performs upsampling using a transposed convolution and then applies two convolutional layers.
    It also includes a skip connection to concatenate feature maps from the downsampling path.
    """

    def __init__(self, in_ch, out_ch):
        """
        Initializes the UpBlock.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """

        # Defining standard convolution parameters
        kernel_size = 3  # 3x3 convolution kernel
        stride = 1  # Stride of 1 to preserve spatial dimensions
        padding = 1  # Padding of 1 to maintain the feature map size

        # Defining transposed convolution parameters
        strideT = 2  # Stride of 2 for doubling the spatial dimensions
        out_paddingT = 1  # Extra padding to align output size correctly

        super().__init__()

        # Define the sequence of layers for the upsampling block
        layers = [
            # Transposed convolution: increases spatial dimensions by a factor of 2
            nn.ConvTranspose2d(2 * in_ch, out_ch, kernel_size, strideT, padding, out_paddingT),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),

            # Standard convolution: processes the upsampled features
            nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        ]

        # Store the layers as a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        """
        Forward pass of the UpBlock.

        Args:
            x (Tensor): Input feature map from the previous layer (upsampled).
            skip (Tensor): Feature map from the downsampling path (skip connection).

        Returns:
            Tensor: Feature map after upsampling and processing.
        """

        # Concatenating the upsampled feature map (x) with the corresponding downsampled feature map (skip connection)
        x = torch.cat((x, skip), dim=1)  # Concatenate along the channel dimension
        return self.model(x)  # Pass through the upsampling layers

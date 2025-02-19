import torch.nn as nn
from unet_blocks import DownBlock, UpBlock

# Defining image and batch parameters
IMG_SIZE = 16
IMG_CH = 1
BATCH_SIZE = 128

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        img_ch = IMG_CH
        down_chs = (16, 32, 64)  # Channel sizes for downsampling layers
        up_chs = down_chs[::-1]  # Reverse order for upsampling layers
        latent_image_size = IMG_SIZE // 4  # Size of the feature map after downsampling

        # Initial convolutional layer
        self.down0 = nn.Sequential(
            nn.Conv2d(img_ch, down_chs[0], 3, padding=1),  # Convolution with 3x3 kernel
            nn.BatchNorm2d(down_chs[0]),  # Normalization for stable training
            nn.ReLU()
        )

        # Downsampling layers
        self.down1 = DownBlock(down_chs[0], down_chs[1])  # First down block
        self.down2 = DownBlock(down_chs[1], down_chs[2])  # Second down block
        self.to_vec = nn.Sequential(nn.Flatten(), nn.ReLU())  # Flatten feature maps into a vector

        # Embedding layers for feature transformation
        self.dense_emb = nn.Sequential(
            nn.Linear(down_chs[2] * latent_image_size ** 2, down_chs[1]),
            nn.ReLU(),
            nn.Linear(down_chs[1], down_chs[1]),
            nn.ReLU(),
            nn.Linear(down_chs[1], down_chs[2] * latent_image_size ** 2),
            nn.ReLU()
        )

        # Upsampling layers
        self.up0 = nn.Sequential(
            nn.Unflatten(1, (up_chs[0], latent_image_size, latent_image_size)),  # Reshape vector back to feature map
            nn.Conv2d(up_chs[0], up_chs[0], 3, padding=1),  # Convolution operation
            nn.BatchNorm2d(up_chs[0]),
            nn.ReLU(),
        )
        self.up1 = UpBlock(up_chs[0], up_chs[1])  # First upsampling block
        self.up2 = UpBlock(up_chs[1], up_chs[2])  # Second upsampling block

        # Output layer to match input channels
        self.out = nn.Sequential(
            nn.Conv2d(up_chs[-1], up_chs[-1], 3, 1, 1),  # Convolution with 3x3 kernel
            nn.BatchNorm2d(up_chs[-1]),
            nn.ReLU(),
            nn.Conv2d(up_chs[-1], img_ch, 3, 1, 1),  # Final convolution to match original image channels
        )

    def forward(self, x):
        # Encoding (downsampling)
        down0 = self.down0(x)  # First convolutional layer
        down1 = self.down1(down0)  # First down block
        down2 = self.down2(down1)  # Second down block
        latent_vec = self.to_vec(down2)  # Flatten feature map

        # Decoding (upsampling)
        up0 = self.up0(latent_vec)  # Reshape vector to feature map
        up1 = self.up1(up0, down2)  # First up block (skip connection with down2)
        up2 = self.up2(up1, down1)  # Second up block (skip connection with down1)
        return self.out(up2)  # Output reconstructed image
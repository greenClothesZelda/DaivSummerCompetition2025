import torch
import torch.nn as nn
from models.modules.conv_block import ConvBlock

class InceptionC(nn.Module):
    def __init__(self, scale, in_channels=2144):
        super().__init__()
        self.scale = scale

        # input 8*8*in_channels, output 8*8*in_channels
        self.branch1_1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=192, kernel_size=1, stride=1, padding=0),  # output: 8x8x192
        )

        self.branch1_2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=192, kernel_size=1, stride=1, padding=0),  # output: 8x8x192
            ConvBlock(in_channels=192, out_channels=224, kernel_size=(1, 3), stride=1, padding=(0, 1)),  # output: 8x8x224
            ConvBlock(in_channels=224, out_channels=256, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # output: 8x8x256
        )

        self.summation = nn.Conv2d(in_channels=448, out_channels=in_channels, kernel_size=1, stride=1, padding=0)  # inception preserve the channel dim

    def forward(self, x):
        x1 = self.branch1_1(x)
        x2 = self.branch1_2(x)

        # Concatenate along the channel dimension
        outputs = [x1, x2]
        inception_out = torch.cat(outputs, dim=1)
        x = x + self.scale * self.summation(inception_out)

        return x

if __name__ == "__main__":
    in_channels = 2144
    inception_c = InceptionC(scale=0.1, in_channels=in_channels)
    x = torch.randn(1, in_channels, 8, 8)  # Batch size of 1, 2144 channels, 8x8 feature map
    output = inception_c(x)
    print(output.shape)  # Should print the shape of the output tensor
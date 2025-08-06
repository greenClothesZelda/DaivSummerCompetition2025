import torch
import torch.nn as nn
from models.modules.conv_block import ConvBlock

class InceptionA(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        # input 35x35x384, output 35x35x32
        self.branch1_1 = nn.Sequential(
            ConvBlock(in_channels=384, out_channels=32, kernel_size=1, stride=1, padding=0),  # output: 35x35x32
        )

        # input 35x35x384, output 35x35x32
        self.branch1_2 = nn.Sequential(
            ConvBlock(in_channels=384, out_channels=32, kernel_size=1, stride=1, padding=0),  # output: 35x35x32
            ConvBlock(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=(1, 1)),  # output: 35x35x32
        )

        # input 35x35x384, output 35x35x64
        self.branch1_3 = nn.Sequential(
            ConvBlock(in_channels=384, out_channels=32, kernel_size=1, stride=1, padding=0),  # output: 35x35x32
            ConvBlock(in_channels=32, out_channels=48, kernel_size=(3, 3), stride=1, padding=(1, 1)),  # output: 35x35x48
            ConvBlock(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=1, padding=(1, 1)),  # output: 35x35x64
        )

        self.branch2 = nn.Conv2d(in_channels=128, out_channels=384, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.branch1_1(x)
        x2 = self.branch1_2(x)
        x3 = self.branch1_3(x)

        # Concatenate along the channel dimension
        outputs = [x1, x2, x3]
        inception_out = torch.cat(outputs, dim=1)
        x = x+ self.scale *self.branch2(inception_out)

        return x

if __name__ == "__main__":
    inception_a = InceptionA(scale=0.1)
    x = torch.randn(1, 384, 35, 35)  # Batch size of 1, 384 channels, 35x35 feature map
    output = inception_a(x)
    print(output.shape)  # Should print the shape of the output tensor
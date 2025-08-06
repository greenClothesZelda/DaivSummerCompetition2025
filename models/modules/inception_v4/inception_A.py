import torch
import torch.nn as nn
from models.modules.conv_block import ConvBlock

class InceptionA(nn.Module):
    def __init__(self):
        super().__init__()
        # input 35x35x384, output 35x35x384
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1), # output: 35x35x384
            ConvBlock(in_channels=384, out_channels=96, kernel_size=1, stride=1, padding=0), # output: 35x35x96
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=384, out_channels=96, kernel_size=1, stride=1, padding=0),
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=384, out_channels=64, kernel_size=1, stride=1, padding=0), # output: 35x35x64
            ConvBlock(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1), # output: 35x35x96
        )

        self.branch4 = nn.Sequential(
            ConvBlock(in_channels=384, out_channels=64, kernel_size=1, stride=1, padding=0), # output: 35x35x64
            ConvBlock(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=1, padding=(1, 1)), # output: 35x35x96
            ConvBlock(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=1, padding=(1, 1)), # output: 35x35x96
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        # Concatenate along the channel dimension
        outputs = [x1, x2, x3, x4]
        x = torch.cat(outputs, dim=1)
        return x

if __name__ == "__main__":
    inception_a = InceptionA()
    x = torch.randn(1, 384, 35, 35)  # Batch size of 1, 384 channels, 35x35 feature map
    output = inception_a(x)
    print(output.shape)  # Should print the shape of the output tensor

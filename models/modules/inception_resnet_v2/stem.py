import torch.nn as nn
import torch

from models.modules.conv_block import ConvBlock
class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        #input size 299*299*3, output size : 35x35x384
        self.feature1 = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0), #output: 149x149x64
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0), #output: 147x147*32
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), #output: 147x147*64
        )
        self.feature2_1 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0), #output: 73x73*96
        )
        self.feature2_2 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=0), #output: 73x73*96
        )

        #input size 73*73*160
        self.feature3_1 = nn.Sequential(
            ConvBlock(in_channels=160, out_channels=64, kernel_size=1, stride=1, padding=0), #output: 73x73*64
            ConvBlock(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=0), #output: 71x71*96
        )

        #input size 73*73*160
        self.feature3_2 = nn.Sequential(
            ConvBlock(in_channels=160, out_channels=64, kernel_size=1, stride=1, padding=0), #output: 73x73*64
            ConvBlock(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3,0)), #output: 73x73*64
            ConvBlock(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0,3)), #output: 73x73*64
            ConvBlock(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=1, padding=0), #output: 71x71*96
        )
        #input size 71*71*192
        self.feature4_1 = nn.Sequential(
            ConvBlock(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=0), #output: 35x35*192
        )
        #input size 71*71*192
        self.feature4_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # output: 35x35*192
        )

    def forward(self, x):
        x = self.feature1(x)
        #concatenate feature2_1 and feature2_2
        x = torch.cat([self.feature2_1(x), self.feature2_2(x)], dim=1)
        #concatenate feature3_1 and feature3_2
        x = torch.cat([self.feature3_1(x), self.feature3_2(x)], dim=1)
        #concatenate feature4_1 and feature4_2
        x = torch.cat([self.feature4_1(x), self.feature4_2(x)], dim=1)
        return x


if __name__ == "__main__":
    stem = Stem()
    x = torch.randn(1, 3, 299, 299)  # Batch size of 1, 3 channels, 299x299 image
    output = stem(x)
    print(output.shape)  # Should print the shape of the output tensor

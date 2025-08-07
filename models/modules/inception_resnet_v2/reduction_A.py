import torch
import torch.nn as nn
from models.modules.conv_block import ConvBlock

class ReductionA(nn.Module):
    def __init__(self, k, l, m, n, in_channels=384):
        super().__init__()
        # input 35x35xin_channels, output 17x17x1024
        self.k = k
        self.l = l
        self.m = m
        self.n = n

        # input 35x35x384, output 17x17x384
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),  # output: 17x17x384
        )

        # input 35x35x384, output 17x17xn
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=n, kernel_size=3, stride=2, padding=0), # output: 17x17xn
        )

        # input 35x35x384, output 17x17xm
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=k, kernel_size=1, stride=1, padding=0), # output: 35x35xk
            ConvBlock(in_channels=k, out_channels=l, kernel_size=(3, 3), stride=1, padding=(1, 1)), # output: 35x35xl
            ConvBlock(in_channels=l, out_channels=m, kernel_size=(3, 3), stride=2, padding=0), # output: 17x17xm
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # Concatenate along the channel dimension
        outputs = [x1, x2, x3]
        x = torch.cat(outputs, dim=1)
        return x

if __name__ == "__main__":
    in_channel = 384
    reduction_a = ReductionA(k=256, l=256, m=384, n=384, in_channels=in_channel)
    x = torch.randn(1, in_channel, 35, 35)  # Batch size of 1, 384 channels, 35x35 feature map
    output = reduction_a(x)
    print(output.shape)
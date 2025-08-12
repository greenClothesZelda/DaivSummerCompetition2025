import torch
import torch.nn as nn
from models.modules.conv_block import ConvBlock

class InceptionResNetA(nn.Module):
    def __init__(self, in_channels, scale=0.1):
        super(InceptionResNetA, self).__init__()

        # Branch 1: 1x1 Conv
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=1, stride=1, padding=0)
        )

        # Branch 2: 1x1 Conv -> 3x3 Conv
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=1, stride=1, padding=0),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1)
        )

        # Branch 3: 1x1 Conv -> 3x3 Conv -> 3x3 Conv
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=1, stride=1, padding=0),
            ConvBlock(32, 48, kernel_size=3, stride=1, padding=1),
            ConvBlock(48, 64, kernel_size=3, stride=1, padding=1)
        )

        # Concatenate output channels: 32 + 32 + 64 = 128
        self.conv_linear = nn.Sequential(
            nn.Conv2d(128, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        out = torch.cat([branch1, branch2, branch3], dim=1)
        out = self.conv_linear(out)

        # Residual scaling
        out = residual + self.scale * out
        out = self.relu(out)

        return out

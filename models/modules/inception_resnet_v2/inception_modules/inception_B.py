import torch
import torch.nn as nn
from models.modules.conv_block import ConvBlock

class InceptionB(nn.Module):
    def __init__(self):
        super().__init__()
        # input 17x17x1024, output 17x17x192
        self.branch1_1 = nn.Sequential(
            ConvBlock(in_channels=1024, out_channels=192, kernel_size=1, stride=1, padding=0),  # output: 17x17x192
        )
        # input 17x17x1024, output 17x17x192
        self.branch1_2 = nn.Sequential(
            ConvBlock(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0),  # output: 17x17x128
            ConvBlock(in_channels=128, out_channels=160, kernel_size=(1, 7), stride=1, padding=(0, 3)),  # output: 17x17x160
            ConvBlock(in_channels=160, out_channels=192, kernel_size=(7, 1), stride=1, padding=(3, 0)),  # output: 17x17x192
        )


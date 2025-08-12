import torch
import torch.nn as nn
from models.modules.conv_block import ConvBlock

class ReductionB(nn.Module):
    def __init__(self, in_channels=896):
        super().__init__()
        #input 17*17*in_channels, output 8*8*1792

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # output: 8x8*in_channels


        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=256, kernel_size=1, stride=1, padding=0),  # output: 17x17*256
            ConvBlock(in_channels=256, out_channels=384, kernel_size=3, stride=2, padding=0)  # output: 8x8*384
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=256, kernel_size=1, stride=1, padding=0),  # output: 17x17*256
            ConvBlock(in_channels=256, out_channels=288, kernel_size=3, stride=2, padding=0),  # output: 8x8*288
        )

        self.branch4 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=256, kernel_size=1, stride=1, padding=0),  # output: 17x17*256
            ConvBlock(in_channels=256, out_channels=288, kernel_size=3, stride=1, padding=1),  # output: 17x17*288
            ConvBlock(in_channels=288, out_channels=320, kernel_size=3, stride=2, padding=0) # output: 8x8*320
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
    in_channels = 192
    reduction_b = ReductionB(in_channels=in_channels)
    x = torch.randn(1, in_channels, 16, 16)  # Batch size of 1, 1024 channels, 17x17 feature map
    output = reduction_b(x)
    print(output.shape)  # Should print the shape of the output tensor

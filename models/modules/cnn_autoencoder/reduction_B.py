from torch import nn
import torch
from models.modules.conv_block import ConvBlock

class ReductionB(nn.Module):
    def __init__(self, in_channels=192):
        super().__init__()
        #input 16*16*192 output 8*8*256

        self.branch1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=0),  # output: 8x8*192
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0),  # output: 16x16*256
            ConvBlock(in_channels=64, out_channels=64, kernel_size=(6, 1), stride=1, padding=(3, 0)),  # output: 16x16*288
            ConvBlock(in_channels=64, out_channels=80, kernel_size=(1, 6), stride=1, padding=(0, 3)),  # output: 16x16*320
            ConvBlock(in_channels=80, out_channels=80, kernel_size=3, stride=2, padding=0)  # output: 8x8*320
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
    in_channels = 192
    reduction_b = ReductionB(in_channels=in_channels)
    x = torch.randn(1, in_channels, 16, 16)  # Batch size of 1, 192 channels, 16x16 feature map
    output = reduction_b(x)
    print(output.shape)  # Should print the shape of the output tensor

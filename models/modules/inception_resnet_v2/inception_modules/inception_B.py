import torch
import torch.nn as nn
from models.modules.conv_block import ConvBlock

class InceptionB(nn.Module):
    def __init__(self, scale = 0.1, in_channels=1152):
        super().__init__()
        self.scale = scale
        # input 17x17x1152, output 17x17x1152 but in paper it is 1154
        #TODO 준석아 한번 검토해봐야해
        self.branch1_1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=192, kernel_size=1, stride=1, padding=0),  # output: 17x17x192
        )

        # input 17x17x1024, output 17x17x192
        self.branch1_2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0),  # output: 17x17x128
            ConvBlock(in_channels=128, out_channels=160, kernel_size=(1, 7), stride=1, padding=(0, 3)),  # output: 17x17x160
            ConvBlock(in_channels=160, out_channels=192, kernel_size=(7, 1), stride=1, padding=(3, 0)),  # output: 17x17x192
        )

        self.summation = nn.Conv2d(in_channels=384, out_channels=in_channels, kernel_size=1, stride=1, padding=0)  # inception preserve the channel dim

    def forward(self, x):
        x1 = self.branch1_1(x)
        x2 = self.branch1_2(x)

        # Concatenate along the channel dimension
        outputs = [x1, x2]
        inception_out = torch.cat(outputs, dim=1)
        x = x + self.scale * self.summation(inception_out)
        return x

if __name__ == "__main__":
    in_channels = 192
    inception_b = InceptionB(scale=0.1, in_channels=in_channels)
    x = torch.randn(1, in_channels, 16, 16)  # Batch size of 1, 1152 channels, 17x17 feature map
    output = inception_b(x)
    print(output.shape)  # Should print the shape of the output tensor

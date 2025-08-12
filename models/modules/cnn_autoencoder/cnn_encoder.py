import torch
from torch import nn
from models.modules.conv_block import ConvBlock
from models.modules.inception_resnet_v2.inception_modules import inception_B, inception_C
from .reduction_B import ReductionB

"""
Input: 64×64×3

1) Conv 3×3, stride=2, filters=32   → 32×32×32
2) Conv 3×3, stride=1, filters=32   → 32×32×32
3) Conv 3×3, stride=1, filters=64   → 32×32×64
4) MaxPool 3×3, stride=2            → 16×16×64
5) Conv 1×1, stride=1, filters=80   → 16×16×80
6) Conv 3×3, stride=1, filters=192  → 16×16×192
7) (선택) MaxPool 3×3, stride=2     → 8×8×192 (작은 모델이면 여기 생략)

Output: 16×16×192 (또는 8×8×192)
"""


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            #inception_resnet_v1 style
            ConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 16x16x64
            ConvBlock(in_channels=64, out_channels=80, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=1),
        )
        self.inceptionB = nn.ModuleList([inception_B.InceptionB(in_channels=192, scale=0.1) for _ in range(5)])
        self.reductionB = ReductionB(in_channels=192)

        self.inceptionC =nn.ModuleList([inception_C.InceptionC(in_channels=336, scale=0.1) for _ in range(10)])

        self.avg = nn.AvgPool2d(kernel_size=8, stride=1)

        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.stem(x)
        for inception in self.inceptionB:
            x = inception(x)
        x = self.reductionB(x)
        for inception in self.inceptionC:
            x = inception(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x

if __name__ == "__main__":
    cnn_encoder = CNNEncoder()
    x = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
    output = cnn_encoder(x)
    print(output.shape)  # Should print the shape of the output tensor
import torch
from torch import nn
from models.modules.conv_block import ConvBlock
from models.modules.inception_resnet_v2.inception_modules import inception_B, inception_C
from .reduction_B import ReductionB  # 상대경로로 수정

class CNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # latent vector를 8x8x336 feature map으로 변환
        self.fc = nn.Linear(336, 8 * 8 * 336)
        self.inceptionC = nn.ModuleList([inception_C.InceptionC(in_channels=336, scale=0.1) for _ in range(10)])
        # ReductionB의 역방향: 채널 수를 336 -> 192로 줄이고, spatial size를 8x8 -> 16x16로 업샘플링
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8x8 -> 16x16
            ConvBlock(in_channels=336, out_channels=192, kernel_size=3, stride=1, padding=1)
        )
        self.inceptionB = nn.ModuleList([inception_B.InceptionB(in_channels=192, scale=0.1) for _ in range(5)])
        # stem의 역방향: 16x16x192 -> 64x64x3
        self.stem_decoder = nn.Sequential(
            ConvBlock(in_channels=192, out_channels=80, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16 -> 32x32
            ConvBlock(in_channels=80, out_channels=64, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 이미지 출력 범위 [-1, 1]로 맞춤
        )

    def forward(self, x):
        # latent vector -> 8x8x336
        x = self.fc(x)
        x = x.view(x.size(0), 336, 8, 8)
        for inception in self.inceptionC:
            x = inception(x)
        x = self.up1(x)
        for inception in self.inceptionB:
            x = inception(x)
        x = self.stem_decoder(x)
        return x

if __name__ == "__main__":
    decoder = CNNDecoder()
    z = torch.randn(1, 128)  # latent vector
    output = decoder(z)
    print(output.shape)  # torch.Size([1, 3, 64, 64])


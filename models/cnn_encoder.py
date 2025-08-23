import torch
from torch import nn

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
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionBlockA(nn.Module):
    def __init__(self, stem_out_channels):
        super(InceptionBlockA, self).__init__()
        branch1_out = stem_out_channels // 3  # Output channels for branch1
        self.branch1 = ConvBlock(in_channels=stem_out_channels, out_channels=branch1_out, kernel_size=3, stride=1,padding=1)

        self.branch2 = ConvBlock(in_channels=stem_out_channels, out_channels=stem_out_channels - branch1_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return torch.cat([x1, x2], dim=1)


class InceptionBlockB(nn.Module):
    def __init__(self, bridge_out):
        super(InceptionBlockB, self).__init__()
        self.branch3 = ConvBlock(in_channels=bridge_out, out_channels=bridge_out // 2, kernel_size=1, stride=1,
                                 padding=0)
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels=bridge_out, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            ConvBlock(in_channels=64, out_channels=bridge_out - bridge_out // 2, kernel_size=(1, 3), stride=1,
                      padding=(0, 1))
        )

    def forward(self, x):
        x1 = self.branch3(x)
        x2 = self.branch4(x)
        return torch.cat([x1, x2], dim=1)

class InceptionBlockC(nn.Module):
    def __init__(self, bridge2_out):
        super(InceptionBlockC, self).__init__()
        self.branch1 = ConvBlock(in_channels=bridge2_out, out_channels=bridge2_out // 4, kernel_size=1, stride=1,
                                 padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=bridge2_out, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            ConvBlock(in_channels=64, out_channels=bridge2_out - bridge2_out // 4, kernel_size=(1, 3), stride=1,
                      padding=(0, 1))
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return torch.cat([x1, x2], dim=1)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        stem_out_channels = 24  # Initial output channels for the stem block
        self.stem = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=stem_out_channels*2, kernel_size=2, stride=2, padding=1),# 64x64x3 -> 33x33x12
            ConvBlock(in_channels=stem_out_channels*2, out_channels=stem_out_channels, kernel_size=3, stride=2, padding=1) #33x33x12 -> 17x17x24
        )

        self.inceptionA = nn.ModuleList([InceptionBlockA(stem_out_channels) for _ in range(3)])  # Two Inception blocks

        bridge_out = stem_out_channels * 2
        self.bridge = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # out 9x9x(inception_out_channels)
            nn.Conv2d(in_channels=stem_out_channels, out_channels=bridge_out, kernel_size=1, stride=1, padding=0)
        )

        self.inceptionB = nn.ModuleList([InceptionBlockB(bridge_out) for _ in range(6)])  # Two Inception blocks

        bridge2_out = bridge_out * 2  # Output channels after Inception blocks
        self.bridge2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # out 5x5x(inception_out_channels)
            nn.Conv2d(in_channels=bridge_out, out_channels=bridge2_out, kernel_size=1, stride=1, padding=0)
        )

        self.inceptionC = nn.ModuleList([InceptionBlockC(bridge2_out) for _ in range(9)])  # Two Inception blocks

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.stem(x)

        for inception_block in self.inceptionA:
            x = inception_block(x)
        x = self.bridge(x)

        for inception_block in self.inceptionB:
            x = inception_block(x)

        x = self.bridge2(x)

        for inception_block in self.inceptionC:
            x = inception_block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x



if __name__ == "__main__":
    # Example usage
    model = CNNEncoder(in_channels=3, out_channels=256)
    #print model total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    #print runnable parameters
    runnable_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters (trainable + non-trainable): {runnable_params}')

    x = torch.randn(1, 3, 64, 64)  # Example input tensor with batch size 1 and image size 64x64
    output = model(x)
    print(output.shape)  # Should print torch.Size([1, out_channels])
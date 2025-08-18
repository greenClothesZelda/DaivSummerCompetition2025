import torch
import torch.nn as nn

"""
Linear(latent_dim, 4*4*256) → reshape 4x4x256
ConvTranspose2d(256, 128, kernel=4, stride=2, padding=1) → 8x8x128
ReLU
ConvTranspose2d(128, 64, kernel=4, stride=2, padding=1) → 16x16x64
ReLU
ConvTranspose2d(64, 32, kernel=4, stride=2, padding=1) → 32x32x32
ReLU
ConvTranspose2d(32, 3, kernel=4, stride=2, padding=1) → 64x64x3
Sigmoid (이미지 [0,1]로 출력)
"""

class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 256),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 이미지 [0, 1] 범위로 출력
        )

    def forward(self, x):
        return self.decoder(x)
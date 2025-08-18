import torch
import torch.nn as nn

"""
Conv2d(3, 32, kernel=4, stride=2, padding=1) → 32x32x32
ReLU
Conv2d(32, 64, kernel=4, stride=2, padding=1) → 16x16x64
ReLU
Conv2d(64, 128, kernel=4, stride=2, padding=1) → 8x8x128
ReLU
Conv2d(128, 256, kernel=4, stride=2, padding=1) → 4x4x256
ReLU
Flatten → Linear(4*4*256, latent_dim)
"""

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)
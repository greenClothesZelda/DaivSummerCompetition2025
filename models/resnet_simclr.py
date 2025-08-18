import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError
from models.cnn_encoder import CNNEncoder


class ResNetSimCLR(nn.Module):

    def __init__(self, num_classes, latent_dim=256):
        super(ResNetSimCLR, self).__init__()
        self.backbone = CNNEncoder(in_channels=3, out_channels=latent_dim)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim*2, num_classes)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
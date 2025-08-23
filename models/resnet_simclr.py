import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError
from models.cnn_encoder import CNNEncoder


class ResNetSimCLR(nn.Module):

    def __init__(self, out_dim=200):
        super(ResNetSimCLR, self).__init__()
        #self.backbone = CNNEncoder(in_channels=3, out_channels=latent_dim)
        self.backbone = models.resnet18(pretrained=False)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Projection Head 정의 (MLP: fc -> ReLU -> fc)
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, out_dim)
        )
    def forward(self, x):
        h = self.backbone(x)  # 특징 벡터 (batch, in_features)
        z = self.projection_head(h)  # projection 벡터 (batch, out_dim)
        return z  # h: representation, z: projection

if __name__ == "__main__":
    import torch
    model = ResNetSimCLR(out_dim=200)
    print(model)
    x = torch.randn(2, 3, 64, 64)
    h, z = model(x)
    print(h.shape, z.shape)  # (2, latent_dim), (2, out_dim)
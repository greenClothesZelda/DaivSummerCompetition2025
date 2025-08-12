from torch import nn
import torch

from .modules.cnn_autoencoder.cnn_encoder import CNNEncoder
from .modules.cnn_autoencoder.cnn_decoder import CNNDecoder


class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder()

    def forward(self, x):
        # Encoder: Input image -> Latent vector
        latent_vector = self.encoder(x)

        # Decoder: Latent vector -> Reconstructed image
        reconstructed_image = self.decoder(latent_vector)

        return reconstructed_image, latent_vector


if __name__ == "__main__":
    autoencoder = CNNAutoencoder()
    x = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
    reconstructed_image, latent_vector = autoencoder(x)
    print(reconstructed_image.shape)  # Should print the shape of the reconstructed image
    print(latent_vector.shape)  # Should print the shape of the latent vector
import torch
import torch.nn as nn
from .decoder_block import DecoderBlock

class Decoder(nn.Module):
    """
    The Decoder for the Masked Autoencoder.
    It takes the encoded patches and reconstructs the original image.
    """
    def __init__(
        self,
        num_layers,
        embed_size,
        heads,
        forward_expansion,
        dropout,
        patch_size,
        channels=3,
    ):
        """
        Initializes the Decoder.

        Args:
            num_layers (int): The number of decoder blocks.
            embed_size (int): The embedding size.
            heads (int): The number of attention heads.
            forward_expansion (int): The expansion factor for the feed-forward layers.
            dropout (float): The dropout rate.
            patch_size (int): The size of the image patches.
            channels (int, optional): The number of output channels. Defaults to 3.
        """
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder_pred = nn.Linear(
            embed_size, patch_size ** 2 * channels
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, embed_size)) # Assuming max patches

    def forward(self, x):
        """
        Forward pass for the Decoder.

        Args:
            x (torch.Tensor): The input tensor from the encoder.

        Returns:
            torch.Tensor: The reconstructed image patches.
        """
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]

        for layer in self.layers:
            x = layer(x)

        x = self.decoder_pred(x)
        return x

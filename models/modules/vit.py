import torch
import torch.nn as nn
from .encoder_block import TransformerBlock

class ViT(nn.Module):
    """
    Vision Transformer (ViT) model, which serves as the encoder for the MAE.
    It processes image patches and learns representations using Transformer blocks.
    """
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        num_layers,
        embed_size,
        heads,
        dropout,
        forward_expansion,
        channels=3,
    ):
        """
        Initializes the Vision Transformer (ViT) model.

        Args:
            image_size (int): The size of the input image (height or width).
            patch_size (int): The size of the image patches.
            num_classes (int): The number of output classes.
            num_layers (int): The number of Transformer blocks.
            embed_size (int): The embedding size.
            heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            forward_expansion (int): The expansion factor for the feed-forward layers.
            channels (int, optional): The number of input channels. Defaults to 3.
        """
        super(ViT, self).__init__()
        self.embed_size = embed_size
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.class_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, img, mask=None):
        """
        Forward pass for the Vision Transformer.

        Args:
            img (torch.Tensor): The input image tensor.
            mask (torch.Tensor, optional): An optional mask for the attention layers. Defaults to None.

        Returns:
            torch.Tensor: The output logits from the classification head.
        """
        x = self.patch_embedding(img)
        x = x.flatten(2).transpose(1, 2)
        batch_size, n_patches, _ = x.shape

        cls_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n_patches + 1)]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, x, x, mask)

        x = self.mlp_head(x[:, 0])
        return x

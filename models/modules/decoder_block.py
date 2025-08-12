import torch
import torch.nn as nn
from .attention import SelfAttention

class DecoderBlock(nn.Module):
    """
    A single block of the Transformer decoder.
    This block is similar to the encoder block but can be simplified for the MAE decoder.
    """
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        """
        Initializes the DecoderBlock.

        Args:
            embed_size (int): The embedding size.
            heads (int): The number of attention heads.
            forward_expansion (int): The expansion factor for the feed-forward layer.
            dropout (float): The dropout rate.
        """
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for the DecoderBlock.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor, optional): An optional mask for the attention layer. Defaults to None.

        Returns:
            torch.Tensor: The output of the decoder block.
        """
        attention = self.attention(x, x, x, mask)
        x = self.dropout(self.norm(attention + x))
        forward = self.transformer_block(x)
        out = self.dropout(self.norm(forward + x))
        return out

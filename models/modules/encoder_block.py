import torch
import torch.nn as nn
from .attention import SelfAttention

class TransformerBlock(nn.Module):
    """
    A single block of the Transformer encoder.
    This block consists of a self-attention layer followed by a feed-forward neural network.
    Layer normalization and dropout are applied for regularization.
    """
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        """
        Initializes the TransformerBlock.

        Args:
            embed_size (int): The embedding size.
            heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            forward_expansion (int): The expansion factor for the feed-forward layer.
        """
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        """
        Forward pass for the TransformerBlock.

        Args:
            value (torch.Tensor): The values tensor for the attention layer.
            key (torch.Tensor): The keys tensor for the attention layer.
            query (torch.Tensor): The query tensor for the attention layer.
            mask (torch.Tensor): An optional mask for the attention layer.

        Returns:
            torch.Tensor: The output of the Transformer block.
        """
        attention = self.attention(value, key, query, mask)

        # Add skip connection, followed by layer normalization
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

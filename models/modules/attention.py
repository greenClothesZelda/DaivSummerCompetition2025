import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    A Self-Attention module, as described in "Attention is All You Need".
    This module computes the attention between different positions of the input sequence.
    """
    def __init__(self, embed_size, heads):
        """
        Initializes the SelfAttention module.

        Args:
            embed_size (int): The embedding size of the input.
            heads (int): The number of attention heads.
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """
        Forward pass of the SelfAttention module.

        Args:
            values (torch.Tensor): The values tensor.
            keys (torch.Tensor): The keys tensor.
            query (torch.Tensor): The query tensor.
            mask (torch.Tensor): An optional mask to be applied to the attention scores.

        Returns:
            torch.Tensor: The output of the self-attention mechanism.
        """
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other training example, then sum it up
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after einsum: (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        return out

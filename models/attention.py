import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention over a sequence of embeddings."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # Project each position to query, key, and value in the same dimension
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_length, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Raw similarity: how much each position "looks at" every other position
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.embed_dim)

        # Normalize scores into a probability distribution over keys (per query row)
        weights = F.softmax(scores, dim=-1)

        # Weighted sum of values — each row mixes information from the whole sequence
        output = weights @ V

        return output

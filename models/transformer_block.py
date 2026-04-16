import torch.nn as nn

from models.attention import SelfAttention


class TransformerBlock(nn.Module):
    """One transformer layer: pre-norm self-attention + residual, then pre-norm FFN + residual."""

    def __init__(self, embed_dim):
        super().__init__()
        self.attention = SelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Position-wise feed-forward: expand, nonlinearity, project back
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        # x: (batch_size, seq_length, embed_dim)

        # Sub-block 1: normalize, attend, residual
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm)
        x = x + attn_out

        # Sub-block 2: normalize, FFN, residual
        x_norm = self.norm2(x)
        ff_out = self.feedforward(x_norm)
        x = x + ff_out

        return x

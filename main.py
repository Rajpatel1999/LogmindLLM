import torch

from models.attention import SelfAttention

x = torch.rand(1, 5, 8)
attention = SelfAttention(embed_dim=8)
out = attention(x)
print("Output shape:", out.shape)

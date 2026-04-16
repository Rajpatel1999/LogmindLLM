import torch
import torch.nn as nn

from models.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """Minimal GPT-style decoder: token + learned positional embeddings, stacked blocks, logits."""

    def __init__(self, vocab_size, embed_dim, num_layers, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim

        # Map token ids to vectors; map each position index to a vector (learned absolute positions)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        # Per-position logits over the vocabulary (no softmax here — loss uses log_softmax / CE)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_length) — integer token indices in [0, vocab_size)
        batch_size, seq_length = x.shape

        tok = self.token_embedding(x)

        # One index per time step: [0, 1, ..., seq_length - 1], broadcast across the batch
        positions = torch.arange(seq_length, device=x.device, dtype=torch.long)
        pos = self.position_embedding(positions)
        h = tok + pos.unsqueeze(0)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        logits = self.fc_out(h)
        return logits

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train(model, dataset, epochs=10, lr=1e-3):
    """Train GPT on (input, next-token) windows from ``TextDataset``."""
    vocab_size = model.fc_out.out_features

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x, y in loader:
            # x, y: (batch, seq_length) — next-token targets align with logits per position
            logits = model(x)
            logits = logits.view(-1, vocab_size)
            y = y.view(-1)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches else 0.0
        print(f"Epoch {epoch + 1}/{epochs}  loss: {avg_loss:.4f}")

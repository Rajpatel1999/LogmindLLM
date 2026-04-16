import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_length)

    def __getitem__(self, idx):
        window = self.tokens[idx : idx + self.seq_length + 1]
        x = window[:-1]
        y = window[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

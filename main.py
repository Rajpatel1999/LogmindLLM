import torch

from models.gpt_model import GPTModel
from utils.tokenizer import CharTokenizer

text = "API failed request API timeout error"

tokenizer = CharTokenizer()
tokenizer.build_vocab(text)

tokens = tokenizer.encode(text)
seq_length = len(tokens)
# Position indices are 0 .. seq_length - 1; table must be at least this long
max_seq_length = max(64, seq_length)

x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

model = GPTModel(
    vocab_size=tokenizer.vocab_size,
    embed_dim=16,
    num_layers=2,
    max_seq_length=max_seq_length,
)

logits = model(x)

print("Logits shape:", logits.shape)

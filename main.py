from data.dataset import TextDataset
from inference.generate import generate
from models.gpt_model import GPTModel
from training.train import train
from utils.tokenizer import CharTokenizer

text = "API failed request API timeout error"

tokenizer = CharTokenizer()
tokenizer.build_vocab(text)

tokens = tokenizer.encode(text)
max_seq_length = max(64, len(tokens))
seq_length = 16

dataset = TextDataset(text, tokenizer, seq_length=seq_length)

model = GPTModel(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    num_layers=2,
    max_seq_length=max_seq_length,
)

train(model, dataset, epochs=10, lr=1e-3)

output = generate(model, tokenizer, start_text="API", max_length=20)
print("Generated:", output)

from utils.tokenizer import SimpleTokenizer

text = "API failed request API timeout error"

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(text)

encoded = tokenizer.encode("API failed request")
decoded = tokenizer.decode(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)
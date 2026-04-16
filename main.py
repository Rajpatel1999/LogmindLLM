from utils.tokenizer import CharTokenizer

sample_corpus = "API failed request API timeout error"

tokenizer = CharTokenizer()
tokenizer.build_vocab(sample_corpus)

sample_string = "API failed request"
encoded = tokenizer.encode(sample_string)
decoded = tokenizer.decode(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)

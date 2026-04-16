class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        for word in text.split():
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text):
        return [self.word_to_id[w] for w in text.split() if w in self.word_to_id]

    def decode(self, token_ids):
        return " ".join(self.id_to_word[i] for i in token_ids)

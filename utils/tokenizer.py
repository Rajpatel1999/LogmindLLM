class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        for ch in text:
            if ch not in self.char_to_id:
                self.char_to_id[ch] = self.vocab_size
                self.id_to_char[self.vocab_size] = ch
                self.vocab_size += 1

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, token_ids):
        return "".join(self.id_to_char[i] for i in token_ids)

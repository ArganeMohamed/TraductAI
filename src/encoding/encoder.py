import json
from src.tokenization.tokenization import tokenize

with open("data/vocabulary.json", "r", encoding="utf-8") as f:
    vocab_to_id = json.load(f)


def encode(text):
    tokens = tokenize(text)

    token_ids = []

    for token in tokens:
        if token in vocab_to_id:
            token_ids.append(vocab_to_id[token])
        else:
            token_ids.append(vocab_to_id["<UNK>"])

    return token_ids
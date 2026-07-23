import json
from src.tokenization.tokenization import tokenize

with open("data/vocabulary.json", "r", encoding="utf-8") as f:
    vocab_to_id = json.load(f)


def encode(text, add_special_tokens=False):

    tokens = tokenize(text)

    token_ids = []

    if add_special_tokens:
        token_ids.append(vocab_to_id["<START>"])

    for token in tokens:
        if token in vocab_to_id:
            token_ids.append(vocab_to_id[token])
        else:
            token_ids.append(vocab_to_id["<UNK>"])

    if add_special_tokens:
        token_ids.append(vocab_to_id["<END>"])

    return token_ids
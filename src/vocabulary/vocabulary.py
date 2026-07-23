import json
from src.tokenization.tokenization import tokenize

with open("data/cleaned_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

vocabulary = set()

for sample in dataset:
    vocabulary.update(tokenize(sample["en"]))
    vocabulary.update(tokenize(sample["fr"]))

print(f"Vocabulary size: {len(vocabulary)}")

vocab_to_id = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<START>": 2,
    "<END>": 3
}

for token in sorted(vocabulary):
    vocab_to_id[token] = len(vocab_to_id)

print(f"Vocabulary with special tokens: {len(vocab_to_id)}")

print("\nFirst 20 tokens:")
for i, (token, idx) in enumerate(vocab_to_id.items()):
    print(f"{token} -> {idx}")

    if i == 19:
        break

with open("data/vocabulary.json", "w", encoding="utf-8") as f:
    json.dump(vocab_to_id, f, ensure_ascii=False, indent=4)

print("Vocabulary saved!")
import json

with open("data/encoded_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

en_lengths = []
fr_lengths = []

for sample in dataset:
    en_lengths.append(len(sample["en"]))
    fr_lengths.append(len(sample["fr"]))

print(f"Number of samples: {len(dataset)}")

print(f"Longest English sentence: {max(en_lengths)} tokens")
print(f"Longest French sentence: {max(fr_lengths)} tokens")

print(f"Average English length: {sum(en_lengths) / len(en_lengths):.2f} tokens")
print(f"Average French length: {sum(fr_lengths) / len(fr_lengths):.2f} tokens")
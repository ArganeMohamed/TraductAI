import json
from src.padding.padding import pad_sequence

MAX_LENGTH = 50

with open("data/encoded_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

padded_data = []

for sample in dataset:
    padded_data.append({
        "en": pad_sequence(sample["en"], MAX_LENGTH),
        "fr": pad_sequence(sample["fr"], MAX_LENGTH)
    })

print(padded_data[1000])
print(f"Total padded pairs: {len(padded_data)}")

with open("data/padded_dataset.json", "w", encoding="utf-8") as f:
    json.dump(padded_data, f, ensure_ascii=False, indent=4)

print("Padded dataset saved!")
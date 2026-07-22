import json
from src.encoding.encoder import encode

with open("data/cleaned_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

encoded_data = []

for sample in dataset:
    encoded_data.append({
        "en": encode(sample["en"]),
        "fr": encode(sample["fr"])
    })
    

with open("data/encoded_dataset.json", "w", encoding="utf-8") as f:
    json.dump(encoded_data, f, ensure_ascii=False, indent=4)

print("Encoded dataset saved!")
from datasets import load_dataset
import json

dataset = load_dataset("Helsinki-NLP/opus_books", "en-fr")
cleaned_data = []

for sample in dataset["train"]:
    en = sample["translation"]["en"].strip()
    fr = sample["translation"]["fr"].strip()

    cleaned_data.append({
        "en": en,
        "fr": fr
    })

seen = set()
unique_data = []

for sample in cleaned_data:
    pair = (sample["en"], sample["fr"])

    if pair not in seen:
        seen.add(pair)
        unique_data.append(sample)

print(f"Rows after removing duplicates: {len(unique_data)}")
print(f"Duplicates removed: {len(cleaned_data) - len(unique_data)}")

none_count = 0

for sample in unique_data:
    if sample["en"] is None or sample["fr"] is None:
        none_count += 1

print(f"Sentence pairs with None: {none_count}")

with open("data/cleaned_dataset.json", "w", encoding="utf-8") as f:
    json.dump(unique_data, f, ensure_ascii=False, indent=4)

print("Cleaned dataset saved successfully")
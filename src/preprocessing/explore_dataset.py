from datasets import load_dataset

dataset = load_dataset("Helsinki-NLP/opus_books", "en-fr")
train = dataset["train"]

print(f"Total sentence pairs: {len(train)}")

empty_count = 0

max_en = 0
max_fr = 0

for sample in train:
    en = sample["translation"]["en"].strip()
    fr = sample["translation"]["fr"].strip()

    if en == "" or fr == "":
        empty_count += 1

    max_en = max(max_en, len(en.split()))
    max_fr = max(max_fr, len(fr.split()))

print(f"Empty sentence pairs : {empty_count}")
print(f"Longest English sentence : {max_en} words")
print(f"Longest French sentence  : {max_fr} words")

print("Example:")
print(train[1000])
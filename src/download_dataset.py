from datasets import load_dataset

dataset = load_dataset(
    "Helsinki-NLP/opus_books",
    "en-fr"
)

print(dataset)
from datasets import load_dataset

dataset = load_dataset(
    "Helsinki-NLP/opus-100",
    "en-fr"
)

print(dataset)
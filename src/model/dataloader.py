import json
import torch
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        english = torch.tensor(sample["en"])
        french = torch.tensor(sample["fr"])

        return english, french


if __name__ == "__main__":
    dataset = TranslationDataset("data/padded_dataset.json")

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    english_batch, french_batch = next(iter(dataloader))

    print("English batch shape:", english_batch.shape)
    print("French batch shape:", french_batch.shape)
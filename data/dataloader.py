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

        english = torch.tensor(
            sample["en"],
            dtype=torch.long
        )

        french = torch.tensor(
            sample["fr"],
            dtype=torch.long
        )

        tgt_input = french[:-1]
        tgt_output = french[1:]

        return english, tgt_input, tgt_output

if __name__ == "__main__":
    dataset = TranslationDataset("data/padded_dataset.json")

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    src_batch, tgt_input_batch, tgt_output_batch = next(iter(dataloader))

    print("Source:", src_batch.shape)
    print("Target input:", tgt_input_batch.shape)
    print("Target output:", tgt_output_batch.shape)

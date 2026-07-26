import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class TranslationDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer_path="data/tokenizer.json", max_len=128):
        with open(f"{data_dir}/{split}.en", encoding="utf-8") as f:
            self.src_lines = [line.strip() for line in f]

        with open(f"{data_dir}/{split}.fr", encoding="utf-8") as f:
            self.tgt_lines = [line.strip() for line in f]

        assert len(self.src_lines) == len(self.tgt_lines)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len

        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.sos_id = self.tokenizer.token_to_id("[SOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")

    def __len__(self):
        return len(self.src_lines)

    def _encode(self, text, add_sos):
        ids = self.tokenizer.encode(text).ids
        ids = ids[: self.max_len - 2]

        if add_sos:
            return [self.sos_id] + ids + [self.eos_id]

        return ids + [self.eos_id]

    def __getitem__(self, index):
        src_ids = self._encode(self.src_lines[index], add_sos=False)
        tgt_ids = self._encode(self.tgt_lines[index], add_sos=True)

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_id):
    src_batch, tgt_batch = zip(*batch)

    src_max = max(len(s) for s in src_batch)
    tgt_max = max(len(t) for t in tgt_batch)

    src_padded = torch.full((len(batch), src_max), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, : len(s)] = s
        tgt_padded[i, : len(t)] = t

    return src_padded, tgt_padded


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = TranslationDataset("data", "valid")
    print("Dataset size:", len(dataset))

    loader = DataLoader(
        dataset, batch_size=8, shuffle=True,
        collate_fn=lambda b: collate_fn(b, dataset.pad_id)
    )

    src_batch, tgt_batch = next(iter(loader))
    print("src_batch shape:", src_batch.shape)
    print("tgt_batch shape:", tgt_batch.shape)
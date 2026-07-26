from datasets import load_dataset

MAX_WORDS = 100
MIN_WORDS = 1
MAX_RATIO = 2.5


def is_valid_pair(en, fr):
    if not en or not fr:
        return False

    en_len = len(en.split())
    fr_len = len(fr.split())

    if en_len < MIN_WORDS or fr_len < MIN_WORDS:
        return False

    if en_len > MAX_WORDS or fr_len > MAX_WORDS:
        return False

    ratio = max(en_len, fr_len) / min(en_len, fr_len)

    if ratio > MAX_RATIO:
        return False

    return True


def clean_split(dataset, hf_split_name):
    seen = set()
    pairs = []

    for sample in dataset[hf_split_name]:
        en = sample["translation"]["en"].strip()
        fr = sample["translation"]["fr"].strip()

        if not is_valid_pair(en, fr):
            continue

        key = (en, fr)

        if key in seen:
            continue

        seen.add(key)
        pairs.append((en, fr))

    print(f"{hf_split_name}: kept {len(pairs)} / {len(dataset[hf_split_name])}")

    return pairs


def write_split(pairs, split_name):
    with open(f"data/{split_name}.en", "w", encoding="utf-8") as f_en, \
         open(f"data/{split_name}.fr", "w", encoding="utf-8") as f_fr:

        for en, fr in pairs:
            f_en.write(en + "\n")
            f_fr.write(fr + "\n")

    print(f"Saved data/{split_name}.en and data/{split_name}.fr")


dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")

for hf_split, local_name in [("train", "train"), ("validation", "valid"), ("test", "test")]:
    pairs = clean_split(dataset, hf_split)
    write_split(pairs, local_name)

print("\nAll splits cleaned and saved.")
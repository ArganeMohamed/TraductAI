import torch
import json

from src.model.transformer import Transformer
from src.training.checkpoint import load_checkpoint


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


VOCAB_SIZE = 105319


# Load vocabulary
with open(
    "data/vocabulary.json",
    "r",
    encoding="utf-8"
) as f:
    vocab = json.load(f)


with open(
    "data/id_to_token.json",
    "r",
    encoding="utf-8"
) as f:
    id_to_token = json.load(f)


# Load model
model = Transformer(
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


load_checkpoint(
    "checkpoints/model_epoch_5.pt",
    model,
    optimizer,
    device
)


model.eval()


# TEST SENTENCE
sentence = "i'm going to morocco"


# Convert words to IDs
tokens = sentence.split()

src_ids = [
    vocab.get(word, 1)
    for word in tokens
]


src = torch.tensor(
    src_ids,
    dtype=torch.long
).unsqueeze(0).to(device)


# Decoder starts with <START>
tgt = torch.tensor(
    [[2]],
    dtype=torch.long
).to(device)


with torch.no_grad():

    for _ in range(20):

        output = model(
            src,
            tgt
        )

        next_token = output[:, -1, :].argmax(
            dim=-1
        )

        tgt = torch.cat(
            [
                tgt,
                next_token.unsqueeze(0)
            ],
            dim=1
        )

        if next_token.item() == 3:
            break


# Convert IDs back to words
result = []

for idx in tgt[0].tolist():

    token = id_to_token.get(str(idx))

    if token not in [
        "<START>",
        "<END>",
        "<PAD>"
    ]:
        result.append(token)


print("Input:")
print(sentence)

print("\nTranslation:")
print(" ".join(result))
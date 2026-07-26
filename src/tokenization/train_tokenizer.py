from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.models import BPE

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
VOCAB_SIZE = 32000

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    min_frequency=2,
)


def corpus_iterator():
    for lang in ("en", "fr"):
        with open(f"data/train.{lang}", encoding="utf-8") as f:
            for line in f:
                yield line.strip()


tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)
tokenizer.save("data/tokenizer.json")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

for text in ["I love machine learning.", "J'adore l'apprentissage automatique."]:
    encoded = tokenizer.encode(text)
    print(f"\nOriginal: {text}")
    print(f"Tokens:   {encoded.tokens}")
    print(f"IDs:      {encoded.ids}")
    print(f"Decoded:  {tokenizer.decode(encoded.ids)}")
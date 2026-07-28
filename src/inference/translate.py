import argparse
import torch
from tokenizers import Tokenizer

from src.model.transformer import Transformer

TOKENIZER_PATH = "data/tokenizer.json"
EMBEDDING_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
FF_DIM = 2048


def load_model(checkpoint_path, device):
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    pad_id = tokenizer.token_to_id("[PAD]")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_size = checkpoint["vocab_size"]

    model = Transformer(
        vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS,
        ff_dim=FF_DIM, num_layers=NUM_LAYERS, pad_id=pad_id
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, step {checkpoint['global_step']}, "
          f"best_val_loss {checkpoint['best_val_loss']:.4f}")

    return model, tokenizer


@torch.no_grad()
def beam_search_translate(model, tokenizer, sentence, device, beam_size=4, max_len=100, length_penalty=0.6):
    sos_id = tokenizer.token_to_id("[SOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    src_ids = tokenizer.encode(sentence.strip()).ids + [eos_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    encoder_output, src_padding_mask = model.encode(src)

    beams = [([sos_id], 0.0, False)]

    for _ in range(max_len):
        if all(finished for _, _, finished in beams):
            break

        candidates = []
        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, finished))
                continue

            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode(tgt, encoder_output, src_padding_mask)
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)

            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
            for lp, idx in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                candidates.append((tokens + [idx], score + lp, idx == eos_id))

        def norm_score(c):
            tokens, score, _ = c
            return score / (len(tokens) ** length_penalty)

        candidates.sort(key=norm_score, reverse=True)
        beams = candidates[:beam_size]

    best_tokens, _, _ = max(beams, key=lambda c: c[1] / (len(c[0]) ** length_penalty))
    ids = best_tokens[1:]
    if eos_id in ids:
        ids = ids[: ids.index(eos_id)]

    return tokenizer.decode(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--beam_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device)

    if args.interactive:
        print("Type an English sentence to translate (empty line to quit):")
        while True:
            text = input("> ").strip()
            if not text:
                break
            print(beam_search_translate(model, tokenizer, text, device, beam_size=args.beam_size))
    elif args.text:
        print(beam_search_translate(model, tokenizer, args.text, device, beam_size=args.beam_size))
    else:
        print("Provide --text '...' or --interactive")


if __name__ == "__main__":
    main()
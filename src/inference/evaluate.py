import argparse
import torch
import sacrebleu
from tqdm import tqdm

from src.inference.translate import load_model, beam_search_translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device)

    with open("data/test.en", encoding="utf-8") as f:
        src_lines = [l.strip() for l in f]
    with open("data/test.fr", encoding="utf-8") as f:
        ref_lines = [l.strip() for l in f]

    if args.limit:
        src_lines, ref_lines = src_lines[: args.limit], ref_lines[: args.limit]

    hyps = [
        beam_search_translate(model, tokenizer, src, device, beam_size=args.beam_size)
        for src in tqdm(src_lines, desc="Translating test set")
    ]

    bleu = sacrebleu.corpus_bleu(hyps, [ref_lines])
    print(f"\nBLEU: {bleu.score:.2f}")

    print("\nSample translations:")
    for i in range(min(5, len(src_lines))):
        print(f"  SRC: {src_lines[i]}")
        print(f"  REF: {ref_lines[i]}")
        print(f"  HYP: {hyps[i]}")
        print()


if __name__ == "__main__":
    main()
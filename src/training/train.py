import os
import argparse

import torch
from torch.utils.data import DataLoader

from data.dataloader import TranslationDataset, collate_fn
from src.model.transformer import Transformer
from src.training.loss import get_loss_function
from src.training.optimizer import get_optimizer, get_scheduler
from src.training.trainer import train_step
from src.training.validate import validate
from src.training.checkpoint import save_checkpoint, load_checkpoint


EMBEDDING_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
FF_DIM = 2048

BATCH_SIZE = 32
EPOCHS = 20
WARMUP_STEPS = 4000
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0

LOG_EVERY = 100
SAVE_EVERY_STEPS = 2000
CHECKPOINT_DIR = "checkpoints"

TEST_SENTENCE = "Hello, how are you doing today ?"


@torch.no_grad()
def translate_sentence(model, tokenizer, sentence, device, max_len=50):
    model.eval()

    sos_id = tokenizer.token_to_id("[SOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    src_ids = tokenizer.encode(sentence).ids + [eos_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    tgt_ids = [sos_id]

    for _ in range(max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        output = model(src, tgt)
        next_id = output[0, -1].argmax(dim=-1).item()

        if next_id == eos_id:
            break

        tgt_ids.append(next_id)

    model.train()

    return tokenizer.decode(tgt_ids[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = TranslationDataset("data", "train")
    valid_dataset = TranslationDataset("data", "valid")

    pad_id = train_dataset.pad_id
    vocab_size = train_dataset.tokenizer.get_vocab_size()
    tokenizer = train_dataset.tokenizer

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=lambda b: collate_fn(b, pad_id))
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               collate_fn=lambda b: collate_fn(b, pad_id))

    print(f"Train batches/epoch: {len(train_loader):,} | Valid batches: {len(valid_loader):,}")

    model = Transformer(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS,
                         ff_dim=FF_DIM, num_layers=NUM_LAYERS, pad_id=pad_id).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = get_loss_function(pad_id=pad_id, label_smoothing=LABEL_SMOOTHING)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, embedding_dim=EMBEDDING_DIM, warmup_steps=WARMUP_STEPS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    start_epoch, global_step, best_val_loss = 0, 0, float("inf")

    if args.resume:
        start_epoch, global_step, best_val_loss = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        print(f"Resumed from {args.resume}: epoch {start_epoch}, step {global_step}, best_val_loss {best_val_loss:.4f}")

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        running_loss, running_count = 0.0, 0

        for src, tgt in train_loader:
            loss = train_step(model, optimizer, scaler, criterion, src, tgt, device,
                               grad_clip=GRAD_CLIP, use_amp=(device.type == "cuda"))
            scheduler.step()

            running_loss += loss
            running_count += 1
            global_step += 1

            if global_step % LOG_EVERY == 0:
                print(f"step {global_step} | loss {running_loss/running_count:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
                running_loss, running_count = 0.0, 0

            if global_step % SAVE_EVERY_STEPS == 0:
                val_loss = validate(model, valid_loader, criterion, device)
                print(f"  [val] step {global_step} | loss {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                     best_val_loss, vocab_size, f"{CHECKPOINT_DIR}/best.pt")
                    print(f"  new best checkpoint saved (val_loss={val_loss:.4f})")

                save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                 best_val_loss, vocab_size, f"{CHECKPOINT_DIR}/last.pt")

                translation = translate_sentence(model, tokenizer, TEST_SENTENCE, device)
                print(f"  [sample] EN: {TEST_SENTENCE}")
                print(f"  [sample] FR: {translation}")

        val_loss = validate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch + 1} done | val_loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step,
                             best_val_loss, vocab_size, f"{CHECKPOINT_DIR}/best.pt")
            print("New best checkpoint saved.")

        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step,
                         best_val_loss, vocab_size, f"{CHECKPOINT_DIR}/last.pt")

        translation = translate_sentence(model, tokenizer, TEST_SENTENCE, device)
        print(f"[sample] EN: {TEST_SENTENCE}")
        print(f"[sample] FR: {translation}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
import torch
from torch.utils.data import (
    DataLoader,
    random_split
)

from src.training.validate import validate

from data.dataloader import TranslationDataset

from src.model.transformer import Transformer

from src.training.loss import get_loss_function
from src.training.optimizer import get_optimizer
from src.training.trainer import train_step
from src.training.checkpoint import save_checkpoint, load_checkpoint


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


BATCH_SIZE = 32

# Total epochs you want to reach
EPOCHS = 20

# Resume training
RESUME = True

CHECKPOINT_PATH = "checkpoints/best_model.pt"


SRC_VOCAB_SIZE = 105319
TGT_VOCAB_SIZE = 105319


dataset = TranslationDataset(
    "data/padded_dataset.json"
)


train_size = int(
    0.9 * len(dataset)
)

val_size = len(dataset) - train_size


generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE
).to(device)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Model parameters: {total_params:,}"
)


criterion = get_loss_function()

optimizer = get_optimizer(
    model
)


start_epoch = 0


if RESUME:

    start_epoch, previous_train_loss, previous_validation_loss = load_checkpoint(
        CHECKPOINT_PATH,
        model,
        optimizer,
        device
    )

    print("Checkpoint loaded.")
    print("Resuming from epoch:", start_epoch)
    print("Previous training loss:", previous_train_loss)
    print("Previous validation loss:", previous_validation_loss)



print("Training setup complete.")

best_validation_loss = float("inf")

if RESUME:
    best_validation_loss = previous_validation_loss

for epoch in range(start_epoch, EPOCHS):

    total_loss = 0

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )


    for batch_idx, (src, tgt_input, tgt_output) in enumerate(train_loader):

        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)


        loss = train_step(
            model,
            optimizer,
            criterion,
            src,
            tgt_input,
            tgt_output
        )


        total_loss += loss


        if batch_idx % 100 == 0:

            print(
                f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss:.4f}"
            )


    average_train_loss = total_loss / len(train_loader)


    print(
        f"Training loss: {average_train_loss:.4f}"
    )

    validation_loss = validate(
        model,
        val_loader,
        criterion,
        device
    )

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            average_train_loss,
            validation_loss,
            "checkpoints/best_model.pt"
        )

        print("New best model saved.")

    print(
        f"Validation loss: {validation_loss:.4f}"
    )


    save_checkpoint(
        model,
        optimizer,
        epoch + 1,
        average_train_loss,
        validation_loss,
        f"checkpoints/model_epoch_{epoch + 1}.pt"
    )


    print(
        f"Checkpoint saved: epoch {epoch + 1}"
    )


print("\nTraining complete.")
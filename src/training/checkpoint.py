import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    validation_loss,
    path
):
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss
    }

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    device
):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    validation_loss = checkpoint["validation_loss"]

    return epoch, train_loss, validation_loss
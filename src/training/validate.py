import torch


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            total_loss += loss.item()
            total_batches += 1

    model.train()
    return total_loss / total_batches
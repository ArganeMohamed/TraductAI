import torch


def validate(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    total_loss = 0

    with torch.no_grad():

        for src, tgt_input, tgt_output in dataloader:

            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            output = model(
                src,
                tgt_input
            )

            output = output.reshape(
                -1,
                output.size(-1)
            )

            tgt_output = tgt_output.reshape(-1)

            loss = criterion(
                output,
                tgt_output
            )

            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    return average_loss
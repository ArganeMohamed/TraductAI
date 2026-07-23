import torch


def train_step(
    model,
    optimizer,
    criterion,
    src,
    tgt_input,
    tgt_output
):
    model.train()

    optimizer.zero_grad()

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

    loss.backward()

    optimizer.step()

    return loss.item()
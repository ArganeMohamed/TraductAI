import torch


def train_step(model, optimizer, scaler, criterion, src, tgt, device, grad_clip=1.0, use_amp=True):
    model.train()

    src = src.to(device)
    tgt = tgt.to(device)

    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=use_amp):
        output = model(src, tgt_input)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()
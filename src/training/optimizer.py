import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)


def noam_lr(step, embedding_dim=512, warmup_steps=4000):
    step = max(step, 1)
    return (embedding_dim ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)


def get_scheduler(optimizer, embedding_dim=512, warmup_steps=4000):
    return lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: noam_lr(step, embedding_dim, warmup_steps)
    )


if __name__ == "__main__":
    import torch.nn as nn

    model = nn.Linear(10, 10)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    print(optimizer)
    for step in [1, 100, 4000, 10000]:
        print(f"step {step}: lr multiplier = {noam_lr(step):.6e}")
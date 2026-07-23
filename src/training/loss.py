import torch.nn as nn


def get_loss_function():
    return nn.CrossEntropyLoss(
        ignore_index=0
    )


if __name__ == "__main__":
    criterion = get_loss_function()

    print(criterion)
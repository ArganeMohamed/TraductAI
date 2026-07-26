import torch.nn as nn


def get_loss_function(pad_id=0, label_smoothing=0.1):
    return nn.CrossEntropyLoss(
        ignore_index=pad_id,
        label_smoothing=label_smoothing
    )


if __name__ == "__main__":
    criterion = get_loss_function()
    print(criterion)
import torch.optim as optim

from src.model.transformer import Transformer


def get_optimizer(model, learning_rate=0.0001):
    return optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


if __name__ == "__main__":

    model = Transformer(
        src_vocab_size=50000,
        tgt_vocab_size=60000
    )

    optimizer = get_optimizer(model)

    print(optimizer)
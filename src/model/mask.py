import torch


def generate_square_subsequent_mask(size):

    mask = torch.triu(
        torch.ones(size, size),
        diagonal=1
    )

    mask = mask.bool()

    return mask


if __name__ == "__main__":

    mask = generate_square_subsequent_mask(5)

    print(mask)
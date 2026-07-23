import torch
import torch.nn as nn

from src.model.encoder import EncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        embedding_dim=128,
        num_heads=8,
        ff_dim=512,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


if __name__ == "__main__":
    encoder = TransformerEncoder()

    x = torch.randn(32, 50, 128)

    output = encoder(x)

    print(x.shape)
    print(output.shape)
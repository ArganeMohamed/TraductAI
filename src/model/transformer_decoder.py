import torch
import torch.nn as nn

from src.model.decoder import DecoderBlock


class TransformerDecoder(nn.Module):
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
                DecoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, encoder_output, tgt_mask=None):

        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                tgt_mask
            )

        return x


if __name__ == "__main__":

    decoder = TransformerDecoder()

    french = torch.randn(32, 50, 128)

    encoder_output = torch.randn(32, 50, 128)

    mask = torch.triu(
        torch.ones(50, 50),
        diagonal=1
    ).bool()

    output = decoder(
        french,
        encoder_output,
        mask
    )

    print(output.shape)
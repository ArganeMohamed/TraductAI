import torch
import torch.nn as nn

from src.model.embedding import Embedding
from src.model.positional_encoding import PositionalEncoding
from src.model.transformer_encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=8,
        ff_dim=512,
        num_layers=6,
    ):
        super().__init__()

        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim
        )

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.positional_encoding(x)

        x = self.encoder(x)

        return x

if __name__ == "__main__":
    model = Transformer(vocab_size=100000)

    x = torch.randint(0, 100000, (32, 50))

    output = model(x)

    print(x.shape)
    print(output.shape)
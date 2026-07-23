import torch
import torch.nn as nn

from src.model.embedding import Embedding
from src.model.positional_encoding import PositionalEncoding
from src.model.transformer_encoder import TransformerEncoder
from src.model.transformer_decoder import TransformerDecoder

from src.model.mask import generate_square_subsequent_mask


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embedding_dim=128,
        num_heads=8,
        ff_dim=512,
        num_layers=6,
    ):
        super().__init__()

        self.src_embedding = Embedding(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim
        )

        self.tgt_embedding = Embedding(
            vocab_size=tgt_vocab_size,
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

        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim
        )

        self.output_layer = nn.Linear(
            embedding_dim,
            tgt_vocab_size
        )

    def forward(self, src, tgt):

        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt == 0)

        src = self.src_embedding(src)
        src = self.positional_encoding(src)

        encoder_output = self.encoder(
            src,
            src_padding_mask
        )


        tgt = self.tgt_embedding(tgt)
        tgt = self.positional_encoding(tgt)

        tgt_mask = generate_square_subsequent_mask(
            tgt.size(1)
        ).to(tgt.device)


        decoder_output = self.decoder(
            tgt,
            encoder_output,
            tgt_mask,
            tgt_padding_mask,
            src_padding_mask
        )

        output = self.output_layer(decoder_output)

        return output

if __name__ == "__main__":
    model = Transformer(
        src_vocab_size=50000,
        tgt_vocab_size=60000
    )

    src = torch.randint(
        0,
        50000,
        (32,50)
    )

    tgt = torch.randint(
        0,
        60000,
        (32,50)
    )

    output = model(src, tgt)

    print(output.shape)
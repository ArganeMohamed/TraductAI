import torch
import torch.nn as nn

from src.model.embedding import Embedding
from src.model.positional_encoding import PositionalEncoding
from src.model.transformer_encoder import TransformerEncoder
from src.model.transformer_decoder import TransformerDecoder
from src.model.mask import generate_square_subsequent_mask


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, num_heads=8,
                 ff_dim=2048, num_layers=6, pad_id=0):
        super().__init__()
        self.pad_id = pad_id

        self.embedding = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim=embedding_dim)

        self.encoder = TransformerEncoder(
            num_layers=num_layers, embedding_dim=embedding_dim,
            num_heads=num_heads, ff_dim=ff_dim
        )
        self.decoder = TransformerDecoder(
            num_layers=num_layers, embedding_dim=embedding_dim,
            num_heads=num_heads, ff_dim=ff_dim
        )

        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        self.output_layer.weight = self.embedding.embedding.weight  # weight tying

    def forward(self, src, tgt):
        src_padding_mask = (src == self.pad_id)
        tgt_padding_mask = (tgt == self.pad_id)

        src_embedded = self.positional_encoding(self.embedding(src))
        encoder_output = self.encoder(src_embedded, src_key_padding_mask=src_padding_mask)

        tgt_embedded = self.positional_encoding(self.embedding(tgt))
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        decoder_output = self.decoder(
            tgt_embedded, encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        return self.output_layer(decoder_output)


if __name__ == "__main__":
    model = Transformer(vocab_size=32000)

    src = torch.randint(0, 32000, (32, 50))
    tgt = torch.randint(0, 32000, (32, 50))

    output = model(src, tgt)
    print(output.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
import torch
import torch.nn as nn

from src.model.encoder import EncoderBlock

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, embedding_dim=512, num_heads=8, ff_dim=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, ff_dim=ff_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return self.norm(x)


if __name__ == "__main__":
    encoder = TransformerEncoder()
    x = torch.randn(32, 50, 512)
    output = encoder(x)

    print(x.shape)
    print(output.shape)
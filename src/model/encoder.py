import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim)
        )

    def forward(self, x, src_key_padding_mask=None):
        normed = self.norm1(x)
        attention_output, _ = self.attention(normed, normed, normed, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(attention_output)

        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)

        return x


if __name__ == "__main__":
    encoder = EncoderBlock()
    x = torch.randn(32, 50, 512)
    output = encoder(x)
    print(x.shape)
    print(output.shape)
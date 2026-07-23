import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=8, ff_dim=512):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attention_output)

        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


if __name__ == "__main__":
    encoder = EncoderBlock()

    x = torch.randn(32, 50, 128)

    output = encoder(x)

    print(x.shape)
    print(output.shape)
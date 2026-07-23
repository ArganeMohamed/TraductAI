import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        num_heads=8,
        ff_dim=512
    ):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )


    def forward(self, x, encoder_output, tgt_mask=None):

        self_attention_output, _ = self.self_attention(
            x,
            x,
            x,
            attn_mask=tgt_mask
        )

        x = self.norm1(
            x + self_attention_output
        )

        cross_attention_output, _ = self.cross_attention(
            x,
            encoder_output,
            encoder_output
        )

        x = self.norm2(
            x + cross_attention_output
        )

        ff_output = self.feed_forward(x)

        x = self.norm3(
            x + ff_output
        )

        return x

if __name__ == "__main__":

    decoder = DecoderBlock()

    french_input = torch.randn(32, 50, 128)

    encoder_output = torch.randn(32, 50, 128)

    output = decoder(
        french_input,
        encoder_output
    )

    print(output.shape)
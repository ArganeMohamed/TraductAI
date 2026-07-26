import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim)
        )

    def forward(self, x, encoder_output, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        normed = self.norm1(x)
        self_attention_output, _ = self.self_attention(
            normed, normed, normed, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = x + self.dropout(self_attention_output)

        normed = self.norm2(x)
        cross_attention_output, _ = self.cross_attention(
            normed, encoder_output, encoder_output, key_padding_mask=memory_key_padding_mask
        )
        x = x + self.dropout(cross_attention_output)

        normed = self.norm3(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)

        return x


if __name__ == "__main__":
    decoder = DecoderBlock()
    french_input = torch.randn(32, 50, 512)
    encoder_output = torch.randn(32, 50, 512)
    output = decoder(french_input, encoder_output)
    print(output.shape)
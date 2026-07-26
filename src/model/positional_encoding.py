import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) *
            (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)


if __name__ == "__main__":
    pe = PositionalEncoding(512)
    x = torch.randn(32, 50, 512)
    output = pe(x)
    print(x[0][0][:10])
    print(output[0][0][:10])
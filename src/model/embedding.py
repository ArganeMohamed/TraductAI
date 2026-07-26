import math
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_dim)


if __name__ == "__main__":
    embedding = Embedding(vocab_size=32000)

    token_ids = torch.tensor([44, 3122, 7228])

    vectors = embedding(token_ids)

    print("Input shape:")
    print(token_ids.shape)

    print("\nEmbedding output shape:")
    print(vectors.shape)

    print("\nFirst vector:")
    print(vectors[0])
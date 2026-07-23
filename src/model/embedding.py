import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.embedding(x)


if __name__ == "__main__":
    embedding = Embedding(vocab_size=105317)

    token_ids = torch.tensor([17099, 54062, 101949])

    vectors = embedding(token_ids)

    print("Input shape:")
    print(token_ids.shape)

    print("\nEmbedding output shape:")
    print(vectors.shape)

    print("\nFirst vector:")
    print(vectors[0])
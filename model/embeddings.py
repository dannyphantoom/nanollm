import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_length: int, embedding_dim: int):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_length, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len)
        return self.positional_embedding[:, :x.size(1), :]


class Embedder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int = 2048, use_positional: bool = True):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.use_positional = use_positional
        self.embedding_dim = embedding_dim

        if self.use_positional:
            self.positional_embedding = PositionalEmbedding(max_length, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok_embed = self.token_embedding(x)  # (B, T, D)
        if self.use_positional:
            pos_embed = self.positional_embedding(x)
            return tok_embed + pos_embed
        return tok_embed


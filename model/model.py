import torch
import torch.nn as nn
from .embeddings import Embedder
from .transformer import TransformerBlock


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_heads: int, n_layers: int,
                 max_seq_len: int = 2048, dropout: float = 0.0, tie_weights: bool = True):
        super().__init__()

        self.embedder = Embedder(vocab_size, dim, max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, seq_len=max_seq_len, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedder.token_embedding.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len) → token IDs
        returns: (batch_size, seq_len, vocab_size) → logits
        """
        x = self.embedder(x)  # (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits


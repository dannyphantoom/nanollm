import torch
import math


def build_rope_cache(seq_len: int, dim: int, base: int = 10000):
    """Precomputes RoPE rotation matrices for a given sequence length and dimension."""
    assert dim % 2 == 0, "Embedding dimension must be even for RoPE."

    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))

    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim/2)

    # Precompute cos and sin matrices
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
    cos = torch.cos(emb)[None, :, None, :]  # shape (1, seq_len, 1, dim)
    sin = torch.sin(emb)[None, :, None, :]  # shape (1, seq_len, 1, dim)

    return cos, sin  # To be applied to q and k

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates every other pair: [x1, x2, x3, x4] → [-x2, x1, -x4, x3]"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).reshape_as(x)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies RoPE to input tensor x.
    x: (batch, seq_len, heads, head_dim)
    cos/sin: (1, seq_len, 1, head_dim)
    """
    return (x * cos) + (rotate_half(x) * sin)


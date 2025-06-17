import torch
import torch.nn as nn
import math
from .rope import build_rope_cache, apply_rotary_pos_emb
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, seq_len: int, use_flash_attn: bool = True):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.seq_len = seq_len
        self.use_flash_attn = use_flash_attn and FLASH_AVAILABLE

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

        # RoPE positional cache
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._build_rope_cache()

    def _build_rope_cache(self):
        # Build RoPE cache for the actual sequence length we'll use
        cos, sin = build_rope_cache(self.seq_len, self.head_dim)
        self.cos = cos.to(torch.float32)  # shape: (1, seq_len, 1, head_dim)
        self.sin = sin.to(torch.float32)  # shape: (1, seq_len, 1, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, dim)
        B, T, _ = x.shape
        
        # Project to q, k, v
        qkv = self.qkv_proj(x)  # (B, T, 3 * dim)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)  # (B, T, 3, n_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each is (B, T, n_heads, head_dim)

        # Apply RoPE to q and k
        # Ensure we only use as much of the cache as we need
        q = apply_rotary_pos_emb(q, self.cos[:, :T], self.sin[:, :T])
        k = apply_rotary_pos_emb(k, self.cos[:, :T], self.sin[:, :T])

        # Prepare for attention
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_heads, T, head_dim)
        v = v.transpose(1, 2)  # (B, n_heads, T, head_dim)

        if self.use_flash_attn:
            # Flash Attention expects (B, T, H, D)
            q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
            attn_output = flash_attn_func(q, k, v, causal=True)
            attn_output = attn_output.transpose(1, 2)  # (B, H, T, D)
        else:
            # Regular scaled dot-product attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        # Merge heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.out_proj(attn_output)


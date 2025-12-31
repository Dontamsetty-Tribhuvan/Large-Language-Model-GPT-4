import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, factor=1.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv / factor, persistent=False)
        self.cache = {}

    def get(self, seq_len, device, dtype):
        if seq_len not in self.cache:
            t = torch.arange(seq_len, device=device)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.cache[seq_len] = (emb.cos().to(dtype), emb.sin().to(dtype))
        return self.cache[seq_len]

def apply_rope(q, k, cos, sin):
    def rot(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    return (q * cos) + (rot(q) * sin), (k * cos) + (rot(k) * sin)

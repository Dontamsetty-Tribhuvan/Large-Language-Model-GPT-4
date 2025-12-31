import math, torch
import torch.nn as nn
import torch.nn.functional as F
from model.rope import apply_rope

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_head
        self.kh = cfg.n_kv_head
        self.d = cfg.n_embd // cfg.n_head
        self.rep = self.h // self.kh

        self.q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.kv = nn.Linear(cfg.n_embd, 2 * self.kh * self.d, bias=False)
        self.o = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        self.flash = cfg.use_flash and hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, rope=None, cache=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.h, self.d).transpose(1, 2)
        kv = self.kv(x).view(B, T, 2, self.kh, self.d)
        k, v = kv.unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        if rope:
            cos, sin = rope
            q, k = apply_rope(q, k, cos[:T], sin[:T])

        if cache is not None:
            pk, pv, plen = cache
            pk[:, :, plen:plen+T] = k
            pv[:, :, plen:plen+T] = v
            k = pk[:, :, :plen+T]
            v = pv[:, :, :plen+T]
            cache[2] += T

        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
            y = att.softmax(-1) @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(y)

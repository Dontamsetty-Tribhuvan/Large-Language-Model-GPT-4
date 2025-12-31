import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, idx, max_new):
    B = idx.size(0)
    cfg = model.cfg
    device = idx.device

    cache = []
    for _ in model.blocks:
        k = torch.zeros(B, cfg.n_kv_head, max_new + idx.size(1), cfg.n_embd // cfg.n_head, device=device)
        v = torch.zeros_like(k)
        cache.append([k, v, 0])

    for _ in range(max_new):
        logits, _ = model(idx[:, -1:], cache=cache)
        probs = F.softmax(logits[:, -1], -1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], 1)

    return idx

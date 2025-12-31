import torch

def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "norm" not in name:
            decay.append(param)
        else:
            no_decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95)
    )

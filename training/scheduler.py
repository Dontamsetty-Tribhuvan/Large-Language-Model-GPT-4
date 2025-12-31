import math

def cosine_scheduler(step, warmup, total_steps, max_lr):
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

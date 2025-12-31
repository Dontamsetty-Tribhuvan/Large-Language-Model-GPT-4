from flash_attn import flash_attn_func

y = flash_attn_func(
    q.transpose(1,2),
    k.transpose(1,2),
    v.transpose(1,2),
    causal=True
).transpose(1,2)

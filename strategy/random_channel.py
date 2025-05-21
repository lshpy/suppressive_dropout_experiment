import torch
import random

def apply_random_dropout(x, drop_ratio=0.25):
    B, C, H, W = x.shape
    k = int(drop_ratio * C)
    masks = torch.ones_like(x)
    for i in range(B):
        indices = torch.randperm(C)[:k]
        masks[i, indices, :, :] = 0
    return x * masks

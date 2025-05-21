import torch
from utils.suppression_utils import compute_suppressive_score

def apply_suppressive_dropout(x, drop_ratio=0.25):
    """
    x: [B, C, H, W] feature map
    return: suppressive score 기반 dropout 적용된 텐서
    """
    B, C, H, W = x.shape
    scores = compute_suppressive_score(x)
    k = int(drop_ratio * C)

    masks = torch.ones_like(x, device=x.device)
    for i in range(B):
        topk = torch.topk(scores[i], k=k).indices
        masks[i, topk, :, :] = 0

    return x * masks

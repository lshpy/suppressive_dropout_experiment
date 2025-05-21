import torch

def compute_suppressive_score(x, b=1.0, c=0.5):
    B, C, H, W = x.shape
    x_sq = (x ** 2).mean(dim=(2, 3))
    total_energy = x_sq.sum(dim=1, keepdim=True)
    denominator = (1 + b * total_energy) ** (c + 1)

    suppressive_score = torch.zeros_like(x_sq)
    for j in range(C):
        x_j_sq = x_sq[:, j].unsqueeze(1)
        for k in range(C):
            if k == j:
                continue
            x_k = x_sq[:, k].unsqueeze(1)
            term = x_k * x_j_sq / denominator
            suppressive_score[:, j] += term.squeeze(1)
    return suppressive_score

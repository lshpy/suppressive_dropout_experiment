import torch.nn.functional as F

def compute_recovery_loss(original, dropped):
    return F.mse_loss(dropped, original.detach())

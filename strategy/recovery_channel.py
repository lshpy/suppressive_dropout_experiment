from strategy.suppressive_channel import apply_suppressive_dropout
from utils.recovery_loss import compute_recovery_loss

# dropout된 feature를 복원하도록 유도하는 loss 포함

def apply_recovery_dropout(x):
    x_dropped = apply_suppressive_dropout(x)
    return x_dropped

def compute_auxiliary_loss(x_original, x_dropped):
    return compute_recovery_loss(x_original, x_dropped)

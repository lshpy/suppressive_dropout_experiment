from strategy.suppressive_channel import apply_suppressive_dropout
from strategy.gradcam_channel import apply_gradcam_amplify

# 억제된 부분은 dropout, 기여 높은 부분은 amplify

def apply_mixed(x):
    x = apply_suppressive_dropout(x)
    x = apply_gradcam_amplify(x)
    return x

from strategy.suppressive_channel import apply_suppressive_dropout
from strategy.gradcam_channel import apply_gradcam_amplify

# 억제 relevance는 높고, attention은 낮은 영역 제거 (현재는 억제 기준만 반영됨)
def apply_hybrid_drop(x):
    x = apply_suppressive_dropout(x)
    return x

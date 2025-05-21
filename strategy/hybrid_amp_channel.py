from strategy.gradcam_channel import apply_gradcam_amplify

# 억제는 낮고, attention과 relevance가 높은 영역 강조 (현재는 amplify만 반영됨)
def apply_hybrid_amp(x):
    return apply_gradcam_amplify(x)

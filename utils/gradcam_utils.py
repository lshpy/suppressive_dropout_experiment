# Grad-CAM utility (placeholder)

def compute_gradcam_map(x):
    return x.mean(dim=(2, 3), keepdim=True)


# suppressive_dropout_experiment/utils/dropout_utils.py
# General dropout mask generator

def generate_dropout_mask(x, ratio):
    return (x > ratio).float()

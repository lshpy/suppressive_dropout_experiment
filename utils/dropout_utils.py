def generate_dropout_mask(x, ratio):
    return (x > ratio).float()

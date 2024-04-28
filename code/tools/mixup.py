import numpy as np
import torch


def mixup_data(x, y_comp, y_attr, y_obj, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index]
    y_comp_a, y_comp_b = y_comp, y_comp[index]
    y_attr_a, y_attr_b = y_attr, y_attr[index]
    y_obj_a, y_obj_b = y_obj, y_obj[index]
    return mixed_x, y_comp_a, y_comp_b, y_attr_a, y_attr_b, y_obj_a, y_obj_b, lam
    



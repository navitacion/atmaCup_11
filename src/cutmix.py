import numpy as np
import torch

# Reference: https://github.com/hysts/pytorch_cutmix/blob/master/cutmix.py
def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets


class CutMixCriterion:
    def __init__(self, criterion_base):
        self.criterion = criterion_base

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets

        return lam * self.criterion(preds, targets1.view_as(preds)) \
               + (1 - lam) * self.criterion(preds, targets2.view_as(preds))
import numpy as np
import torch


# Reference: https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py#L119
def mixup(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    targets = (y_a, y_b, lam)

    return mixed_x, targets


class MixupCriterion:
    def __init__(self, criterion_base):
        self.criterion = criterion_base

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets

        return lam * self.criterion(preds, targets1.view_as(preds)) \
               + (1 - lam) * self.criterion(preds, targets2.view_as(preds))
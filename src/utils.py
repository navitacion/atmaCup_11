import numpy as np
import torch
from torch import optim
from timm.optim import RAdam

class Converter:
    def __init__(self, ymax, ymin):
        self.ymax = ymax
        self.ymin = ymin

    def transform(self, year):
        # Year -> Continuous value
        cont = (year - self.ymin) / (self.ymax - self.ymin)

        return cont

    def inverse_transform(self, cont):
        # Continuous value -> Year
        year = cont * (self.ymax - self.ymin) + self.ymin

        # Year -> Category
        if year <= 1600:
            return 0.0
        elif 1600 < year <= 1700:
            return 1.0
        elif 1700 < year <= 1800:
            return 2.0
        elif year > 1800:
            return 3.0

# Ref: https://www.guruguru.science/competitions/17/discussions/000d76a9-fc4b-443e-95f2-5c066c0f3108/
class Converter2:
    def __init__(self):
        pass

    def transform(self, year):
        # Year -> Continuous value
        cont = year / 100 - 15.51

        return cont

    def inverse_transform(self, cont):
        # Continuous value -> Category
        return round(cont, 0)


class Converter3:
    def __init__(self, ymax, ymin):
        self.ymax = ymax
        self.ymin = ymin

    def transform(self, year):
        # Year -> Continuous value
        cont = (year - self.ymin) / (self.ymax - self.ymin)

        return cont

    def inverse_transform(self, cont):
        # Continuous value -> Year
        year = cont * (self.ymax - self.ymin) + self.ymin

        # Year -> Category
        if year <= 1550:
            return 0.0
        elif 1550 < year <= 1650:
            return 0.0 + (year - 1550) / (1650 - 1550)
        elif 1650 < year <= 1750:
            return 1.0 + (year - 1650) / (1750 - 1650)
        elif 1750 < year <= 1850:
            return 2.0 + (year - 1750) / (1850 - 1750)
        elif year > 1850:
            return 3.0


class Converter4:
    def __init__(self, ymean, ystd):
        self.ymean = ymean
        self.ystd = ystd

    def transform(self, year):
        # Year -> Continuous value
        cont = (year - self.ymean) / self.ystd

        return cont

    def inverse_transform(self, cont):
        # Continuous value -> Year
        year = cont * self.ystd + self.ymean

        # Year -> Category
        if year <= 1550:
            return 0.0
        elif 1550 < year <= 1650:
            return 0.0 + (year - 1550) / (1650 - 1550)
        elif 1650 < year <= 1750:
            return 1.0 + (year - 1650) / (1750 - 1650)
        elif 1750 < year <= 1850:
            return 2.0 + (year - 1750) / (1850 - 1750)
        elif year > 1850:
            return 3.0


def get_optim(net, cfg):
    if cfg.train.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
    elif cfg.train.optimizer == 'radam':
        optimizer = RAdam(net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    return optimizer

if __name__ == '__main__':

    print('')
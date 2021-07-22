import torchvision

from abc import ABCMeta
import albumentations as albu
from albumentations.pytorch import ToTensorV2


class BaseTransform(metaclass=ABCMeta):
    def __init__(self):
        self.transform = None

    def __call__(self, img, phase='train'):
        transformed_img = self.transform[phase](image=img)['image']

        return transformed_img


class ImageTransform(BaseTransform):
    def __init__(self, cfg):
        super(ImageTransform, self).__init__()
        aug_config = cfg.aug_kwargs

        transform_train_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(aug_config.train).items()]
        transform_train_list.append(ToTensorV2())
        transform_val_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(aug_config.val).items()]
        transform_val_list.append(ToTensorV2())
        transform_test_list = [getattr(albu, name)(**kwargs) for name, kwargs in dict(aug_config.test).items()]
        transform_test_list.append(ToTensorV2())


        self.transform = {
            'train': albu.Compose(transform_train_list, p=1.0),
            'val': albu.Compose(transform_val_list, p=1.0),
            'test': albu.Compose(transform_test_list, p=1.0)
        }
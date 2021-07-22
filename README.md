# atmaCup #11

## URL

https://www.guruguru.science/competitions/17/

## References

- [StratifiedGroupKFold](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [Scikit-Learn Nightly](https://scikit-learn.org/stable/developers/advanced_installation.html#installing-nightly-builds)
- [Albumentations](https://albumentations.ai/docs/)
- [timm](https://github.com/rwightman/pytorch-image-models)



## Result

- Private Score: 
- Rank: th /  (%)


## Getting Started

Easy to do, only type command.

```commandline
docker-compose up --build -d
docker exec -it atma_env bash
```

## Solution

### Cross Validation
- [StratifiedGroupKFold](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
  - y: target, group: art_series_id
  - Using [Scikit-Learn Nightly](https://scikit-learn.org/stable/developers/advanced_installation.html#installing-nightly-builds)

### Augmentations
- based on [Albumentations](https://albumentations.ai/docs/)
    - Good Work: ToGray, HorizontalFlip, VerticalFlip, RandomRotate90, CoarseDropout
    - Bad Work : RandomResizedCrop, Mixup, Cutmix, ColorJitter
    
### Model Network
- Simple CNN Network (based on [timm](https://github.com/rwightman/pytorch-image-models))
    - Resnet34d
    - Resnet18d
- Tried but not useful...
    - Resnet50d
    - Efficientnet-b0

### Classification vs Regression

- I tried only Regression Task
    - Convert `sorting_date` to continuous value
        - Standardization is the best
            - (For some reason min-max normalization didn't work on my env)
        - For submit, output value is 0.0 - 3.0 depending on Year
    
```python
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

```

### Params of best Single Model

- LB: 0.7218
```yaml
data:
  seed: 0
  img_size: &image_size 224
  n_splits: 5
  cv: stratifiedgroup

train:
  lr: 1e-4
  epoch: 160
  batch_size: 16
  num_workers: 8
  backbone: resnet34d
  tta_num: 4
  optimizer: adam
  scheduler: CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=0)

aug_kwargs:
  train:
    Resize: {"height": *image_size, "width": *image_size}
    ToGray: { "p": 0.2 }
    HorizontalFlip: { "p": 0.5 }
    VerticalFlip: { "p": 0.5 }
    RandomRotate90: { "p": 0.5 }
    Normalize: {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "p": 1.0}
    CoarseDropout: {"max_holes": 8, "max_width": 8, "max_height": 8}

  val:
    Resize: {"height": *image_size, "width": *image_size}
    Normalize: {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

  test:
    Resize: {"height": *image_size, "width": *image_size}
    HorizontalFlip: { "p": 0.5 }
    VerticalFlip: { "p": 0.5 }
    Normalize: {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


```

### Ensemble

- Random Seed Average
    - Seed: 0, 1, 2, 3 (Only 4 Model...)


## Model Training

only execute below

```commandline
python train.py
```

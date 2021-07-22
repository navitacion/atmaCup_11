import os
import itertools
import time

import wandb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import metrics

from .dataset import atmaDataset
from .mixup import mixup, MixupCriterion
from .cutmix import cutmix, CutMixCriterion


class AtmaDataModule(pl.LightningDataModule):
    """
    DataModule for Cassava Competition
    """
    def __init__(self, df, test, cfg, transform, converter=None):
        """
        ------------------------------------
        Parameters
        cfg: DictConfig
            Config
        transform: albumentations.transform
            Data Augumentations
        """
        super(AtmaDataModule, self).__init__()
        self.df = df
        self.test = test
        self.cfg = cfg
        self.transform = transform

    def prepare_data(self):
        # Merge 'materials.csv'
        materials = pd.read_csv(os.path.join(self.cfg.data.data_dir, 'materials.csv'))
        # 複数の変数がある場合は結合
        materials = materials.groupby('object_id')['name'].apply(lambda x: '_'.join(sorted(list(x)))).reset_index()
        self.df = pd.merge(self.df, materials, on='object_id', how='left')
        lbl = LabelEncoder()
        self.df['name'] = lbl.fit_transform(self.df['name'])

        # Cross Validation
        if self.cfg.data.cv == 'stratified':
            cv = StratifiedKFold(
                n_splits=self.cfg.data.n_splits,
                shuffle=True,
                random_state=self.cfg.data.seed
            )
            self.df['fold'] = -1
            for i, (trn_idx, val_idx) in enumerate(cv.split(self.df, self.df['org_target'])):
                self.df.loc[val_idx, 'fold'] = i

        elif self.cfg.data.cv == 'kfold':
            cv = KFold(
                n_splits=self.cfg.data.n_splits,
                shuffle=True,
                random_state=self.cfg.data.seed
            )
            self.df['fold'] = -1
            for i, (trn_idx, val_idx) in enumerate(cv.split(self.df)):
                self.df.loc[val_idx, 'fold'] = i

        elif self.cfg.data.cv == 'stratifiedgroup':
            cv = StratifiedGroupKFold(
                n_splits=self.cfg.data.n_splits,
                shuffle=True,
                random_state=self.cfg.data.seed
            )
            self.df['fold'] = -1
            for i, (trn_idx, val_idx) in enumerate(cv.split(self.df, self.df['org_target'], groups=self.df['art_series_id'])):
                self.df.loc[val_idx, 'fold'] = i
        else:
            self.df['fold'] = 0

    def setup(self, stage=None):
        train = self.df[self.df['fold'] != self.cfg.train.fold].reset_index(drop=True)
        val = self.df[self.df['fold'] == self.cfg.train.fold].reset_index(drop=True)

        # Dataset
        self.train_dataset = atmaDataset(train, self.cfg, self.transform, phase='train')
        self.val_dataset = atmaDataset(val, self.cfg, self.transform, phase='val')
        self.test_dataset = atmaDataset(self.test, self.cfg, self.transform, phase='test')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            pin_memory=True,
            num_workers=self.cfg.train.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.batch_size,
            pin_memory=True,
            num_workers=self.cfg.train.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.train.batch_size,
            pin_memory=True,
            num_workers=self.cfg.train.num_workers,
            shuffle=False
        )



class AtmaLightningSystem(pl.LightningModule):
    def __init__(self, net, cfg, criterion=None, optimizer=None, scheduler=None, converter=None):
        """
        ------------------------------------
        Parameters
        net: torch.nn.Module
            Model
        cfg: DictConfig
            Config
        optimizer: torch.optim
            Optimizer
        scheduler: torch.optim.lr_scheduler
            Learning Rate Scheduler
        """
        super(AtmaLightningSystem, self).__init__()
        self.net = net
        self.cfg = cfg
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.converter = converter
        self.weight_paths = []
        self.best_rmse = 1e+3
        self.best_acc = 0.5

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer], []
        else:
            return [self.optimizer], [self.scheduler]

    def forward(self, x):
        output = self.net(x)
        return output

    def step(self, batch, phase='train'):
        rand = np.random.rand()

        if phase == 'test':
            img, object_id = batch
            out = self.forward(img)
            return out, object_id

        else:
            img, object_id, reg_label, org_label = batch
            # Regression / Classification
            label = reg_label if self.cfg.train.mode == 'regression' else org_label

            # mixup
            if self.cfg.train.mixup_alpha > 0 and \
                    rand > (1.0 - self.cfg.train.mixup_pct) and \
                    (self.cfg.train.epoch * 0.9) > self.current_epoch:  # 学習の最後の方はmixupを行わない
                img, label = mixup(img, label, alpha=self.cfg.train.mixup_alpha)
                out = self.forward(img)
                loss_fn = MixupCriterion(criterion_base=self.criterion)
                loss = loss_fn(out, label)

            # cutmix
            elif self.cfg.train.cutmix_alpha > 0 and \
                    rand > (1.0 - self.cfg.train.cutmix_pct) and \
                    (self.cfg.train.epoch * 0.9) > self.current_epoch:  # 学習の最後の方はcutmixを行わない
                img, label = cutmix(img, label, alpha=self.cfg.train.cutmix_alpha)
                out = self.forward(img)
                loss_fn = CutMixCriterion(criterion_base=self.criterion)
                loss = loss_fn(out, label)
            else:
                out = self.forward(img)

                if self.cfg.train.mode == 'regression':
                    loss = self.criterion(out, label.view_as(out))
                else:
                    loss = self.criterion(out, label)

            return loss, label, org_label, object_id, out

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self.step(batch, phase='train')
        self.log(f'fold{self.cfg.train.fold}_train/loss', loss, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, label, org_label, object_id, out = self.step(batch, phase='val')
        self.log(f'fold{self.cfg.train.fold}_val/loss', loss, on_epoch=True)

        return {'val_loss': loss, 'logits': out, 'labels': label, 'org_labels': org_label, 'object_id': object_id}


    def test_step(self, batch, batch_idx):
        out, object_id = self.step(batch, phase='test')

        return {'logits': out, 'object_id': object_id}

    def validation_epoch_end(self, outputs):

        if self.cfg.train.mode == 'regression':
            # 予測結果と元のラベルのRMSEを計算する
            logits = torch.cat([x['logits'] for x in outputs]).detach().cpu().numpy().reshape((-1))
            org_label = torch.cat([x['org_labels'] for x in outputs]).detach().cpu().numpy().reshape((-1)).tolist()
            convert_preds = [self.converter.inverse_transform(x) for x in logits.tolist()]

            rmse = np.sqrt(mean_squared_error(y_true=org_label, y_pred=convert_preds))

            # Logging
            self.log(f'fold{self.cfg.train.fold}_val/epoch_rmse', rmse, on_step=False, on_epoch=True)

            # Save Weights
            if rmse < self.best_rmse:
                filename = '{}-seed_{}_fold_{}_ims_{}_epoch_{}_rmse_{:.3f}.pth'.format(
                    self.cfg.train.backbone, self.cfg.data.seed, self.cfg.train.fold,
                    self.cfg.data.img_size, self.current_epoch, rmse
                )
                torch.save(self.net.state_dict(), os.path.join(self.cfg.data.checkpoint_path, filename))
                self.weight_paths.append(os.path.join(self.cfg.data.checkpoint_path, filename))

                self.best_rmse = rmse

        else:
            # 予測結果と元のラベルのAccuracyを計算する
            logits = torch.cat([x['logits'] for x in outputs]).detach().cpu()
            org_label = torch.cat([x['org_labels'] for x in outputs]).detach().cpu()
            acc = metrics.Accuracy()(logits, org_label)

            # Logging
            self.log(f'fold{self.cfg.train.fold}_val/epoch_acc', acc, on_step=False, on_epoch=True)

            # Save Weights
            if acc > self.best_acc:
                filename = '{}-seed_{}_fold_{}_ims_{}_epoch_{}_acc_{:.3f}.pth'.format(
                    self.cfg.train.backbone, self.cfg.data.seed, self.cfg.train.fold,
                    self.cfg.data.img_size, self.current_epoch, acc
                )
                torch.save(self.net.state_dict(), os.path.join(self.cfg.data.checkpoint_path, filename))
                self.weight_paths.append(os.path.join(self.cfg.data.checkpoint_path, filename))

                self.best_acc = acc


        return None


    def test_epoch_end(self, outputs):

        logits = torch.cat([x['logits'] for x in outputs]).detach().cpu().numpy().reshape((-1))
        logits = [self.converter.inverse_transform(x) for x in logits.tolist()]

        ids = [x['object_id'] for x in outputs]
        ids = [list(x) for x in ids]
        ids = list(itertools.chain.from_iterable(ids))

        self.res = pd.DataFrame({
            'object_id': ids,
            'target': logits
        })

        test = pd.read_csv(os.path.join(self.cfg.data.data_dir, 'test.csv'))

        self.res = pd.merge(test, self.res, on='object_id', how='left')

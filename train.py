import os
import gc
import glob
import time
import shutil
import pandas as pd
import hydra
from omegaconf import DictConfig
import wandb

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from timm.optim import RAdam
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from src.lightning import AtmaDataModule, AtmaLightningSystem
from src.transform import ImageTransform
from src.model import Timm_model, Torch_model
from src.losses import RMSELoss
from src.utils import Converter, Converter2, Converter3, Converter4, get_optim


@hydra.main(config_name='config.yml')
def main(cfg: DictConfig):
    print('atmaCup #11 Training Script')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)

    os.makedirs(cfg.data.checkpoint_path, exist_ok=True)

    # Config  -------------------------------------------------------------------
    cfg.train.mode = 'regression'
    seed_everything(cfg.data.seed)

    wandb.login()
    wandb_logger = WandbLogger(project='atmaCup-11', reinit=True)
    wandb_logger.log_hyperparams(dict(cfg.data))
    wandb_logger.log_hyperparams(dict(cfg.train))
    wandb_logger.log_hyperparams(dict(cfg.aug_kwargs))

    # Data Module  -------------------------------------------------------------------
    # Prepare Data
    df = pd.read_csv(os.path.join(cfg.data.data_dir, 'train.csv'))
    df['org_target'] = df['target']
    test = pd.read_csv(os.path.join(cfg.data.data_dir, 'test.csv'))

    # Convert Values
    converter = Converter4(ymean=df['sorting_date'].mean(), ystd=df['sorting_date'].std())
    df['target'] = df['sorting_date'].apply(converter.transform)

    transform = ImageTransform(cfg)
    dm = AtmaDataModule(df, test, cfg, transform)

    # Train  ----------------------------------------------------------------------------
    rmse = 0
    cnt = 0
    model_weights = []

    # Train on Fold
    for fold in range(cfg.data.n_splits):
        cfg.train.fold = fold

        # Model  -----------------------------------------------------------
        net = Timm_model(cfg.train.backbone, pretrained=False, out_dim=1)

        # Loss fn  -----------------------------------------------------------
        criterion = RMSELoss()

        # Optimizer, Scheduler  -----------------------------------------------------------
        optimizer = get_optim(net, cfg)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=0)
        # Lightning Module
        model = AtmaLightningSystem(net, cfg, criterion, optimizer, scheduler, converter)

        # Callback  -----------------------------------------------------
        # early_stopping = EarlyStopping(monitor='val/epoch_rmse', patience=10, mode='min')

        # Trainer  --------------------------------------------------------------------------
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=cfg.train.epoch,
            gpus=-1,
            num_sanity_val_steps=0,
            fast_dev_run=False,
            # callbacks=[early_stopping],
            # deterministic=True,
        )

        # Train
        trainer.fit(model, datamodule=dm)
        rmse += model.best_rmse
        model_weights.append(model.weight_paths[-1])
        cnt += 1

        if cfg.train.only_one_fold:
            break

    rmse /= cnt
    wandb.log({'val/best_rmse': rmse})
    for path in model_weights:
        wandb.save(path)

    # Inference  --------------------------------------------------------------------------
    print('Inference')
    all_res = pd.read_csv(os.path.join(cfg.data.data_dir, 'test.csv'))
    model_weights = sorted(model_weights)

    # Inference on Fold
    for fold, weight_path in enumerate(model_weights):
        net = Timm_model(cfg.train.backbone, pretrained=False, out_dim=1)
        net.load_state_dict(torch.load(weight_path))
        net = net.eval()
        model = AtmaLightningSystem(net, cfg, converter=converter)
        trainer = Trainer(logger=False, gpus=-1, num_sanity_val_steps=0)

        for i in range(cfg.train.tta_num):
            trainer.test(model, datamodule=dm, ckpt_path=None)
            if i == 0:
                all_res['target'] = model.res['target']
            else:
                all_res['target'] += model.res['target']

        all_res['target'] /= cfg.train.tta_num
        filename = f'submission_{fold}.csv'
        all_res[['target']].to_csv(filename, index=False)
        wandb.save(filename)

    # concat sub
    subs = glob.glob('submission*')
    all_res = pd.read_csv(os.path.join(cfg.data.data_dir, 'test.csv'))
    all_res['target'] = 0
    for path in subs:
        tmp = pd.read_csv(path)
        all_res['target'] += tmp['target']
    all_res['target'] /= len(subs)
    all_res[['target']].to_csv('submission.csv', index=False)
    wandb.save('submission.csv')

    # Stop Logging
    wandb.finish()
    time.sleep(5)

    # Remove checkpoint folder
    shutil.rmtree(cfg.data.checkpoint_path)
    for path in subs:
        os.remove(path)
    os.remove('submission.csv')

    del model, all_res, net, dm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
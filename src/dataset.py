import os
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class atmaDataset(Dataset):
    def __init__(self, df, cfg=None, transform=None, phase='train'):
        self.df = df
        self.cfg = cfg
        self.transform = transform
        self.phase = phase
        self.img_path = glob.glob(os.path.join(self.cfg.data.data_dir, 'photos', '*.jpg'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target_img_id = row['object_id']
        target_img_path = [path for path in self.img_path if target_img_id in path][0]

        img = cv2.imread(target_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        if self.transform is not None:
            img = self.transform(img, self.phase)
        else:
            img = torch.from_numpy(img.transpose((2, 0, 1)))
            img = img / 255.

        if self.phase == 'test':
            return img, target_img_id

        else:
            label = torch.tensor(row['target'], dtype=torch.float)
            org_label = torch.tensor(row['org_target'], dtype=torch.long)
            return img, target_img_id, label, org_label


if __name__ == '__main__':
    df = pd.read_csv('./input/train.csv')

    dataset = atmaDataset(df, transform=None, phase='train')

    inp = dataset.__getitem__(7)

    print(inp['img'].size())
    print(inp['label'])
    print(inp['id'])
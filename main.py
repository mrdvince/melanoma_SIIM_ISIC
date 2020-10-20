import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import pandas as pd
import pretrainedmodels
import albumentations
from .dataloader import ClassificationDataLoader
from .early_stopping import EarlyStopping


class SEResNext50(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResNext50, self).__init__()
        self.models = pretrainedmodels.__dict__[
            'se_resnext50_32x4d'](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        return out


def train(fold):
    train_input_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 50
    train_bs = 32
    valid_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_aug = albumentations.Compose([
        albumentations.CenterCrop(224, 224, always_apply=True),
        albumentations.Normalize(
            mean, std, max_pixel_value=255, always_apply=True),
    ])

    valid_aug = albumentations.Compose([
        albumentations.CenterCrop(224, 224, always_apply=True),
        albumentations.Normalize(
            mean, std, max_pixel_value=255, always_apply=True),
    ])

    train_images = df_train.image_name.values.toList()
    train_images = [os.path.join(train_input_path, i + '.jpg')
                    for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.toList()
    valid_images = [os.path.join(train_input_path, i + '.jpg')
                    for i in valid_images]
    valid_targets = df_valid.target.values

    train_ds = ClassificationDataLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug
    )

    train_loader = torch.utils.data.dataloader(
        train_ds,
        batch_size=train_bs,
        shuffle=True
    )

    valid_ds = ClassificationDataLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )

    valid_loader = torch.utils.data.dataloader(
        valid_ds,
        batch_size=valid_bs,
        shuffle=False
    )

    model = SEResNext50(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode='max'
    )

    es = EarlyStopping(patience=5)
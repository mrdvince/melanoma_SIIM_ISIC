import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import pandas as pd
import pretrainedmodels
import albumentations


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
    input_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
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

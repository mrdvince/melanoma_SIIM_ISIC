import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import pandas as pd
import pretrainedmodels
import albumentations
from sklearn import metrics
from dataloader import ClassificationLoader
from early_stopping import EarlyStopping
from engine import Engine


class SEResNext50(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResNext50, self).__init__()
        self.models = pretrainedmodels.__dict__[
            'se_resnext50_32x4d'](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.models.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))
        return out, loss


def train(fold):
    train_input_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
    model_path = '/kaggle/working/melanoma-SIIM-ISIC'
    df = pd.read_csv('train_folds.csv')
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

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(train_input_path, i + '.jpg')
                    for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(train_input_path, i + '.jpg')
                    for i in valid_images]
    valid_targets = df_valid.target.values

    train_ds = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=train_bs
    )

    valid_ds = ClassificationLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=valid_bs, shuffle=False, num_workers=4
    )

    model = SEResNext50(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode='max'
    )

    es = EarlyStopping(patience=5, mode='max')

    for epoch in range(epochs):
        train_loss = Engine.train(
            data_loader=train_loader,
            model=model,
            optimizer=optimizer,
            device=device
        )

        predictions, valid_loss = Engine.evaluate(
            data_loader=valid_loader,
            model=model,
            optimizer=optimizer,
            device=device
        )
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)

        print(f'Epoch= {epoch}, auc= {auc}')
        es(auc, model, model_path)
        if es.early_stop:
            print('Early stopping')
            break


if __name__ == '__main__':
    train(fold=0)

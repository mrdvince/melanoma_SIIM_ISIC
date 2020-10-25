

import os
from flask import Flask, request, render_template
import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import pandas as pd
import pretrainedmodels
import albumentations
import sys

sys.path.append('model')
from dataloader import ClassificationLoader
from engine import Engine

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResnext50_32x4d, self).__init__()
        self.models = pretrainedmodels.__dict__[
            'se_resnext50_32x4d'](pretrained=pretrained)
        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.models.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))
        return out, loss


def predict(image_path, model):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )
    images = [image_path]
    targets = [0]
    test_dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=aug,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )
    predictions = Engine.predict(test_loader, model, device=device)
    predictions = np.vstack((predictions)).ravel()
    return predictions


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_file.save(os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            ))
            return render_template('index.html', prediction=1)

    return render_template('index.html', prediction=0)


if __name__ == '__main__':
    MODEL = SEResnext50_32x4d(pretrained=None)
    MODEL.load_state_dict(torch.load(
        'model/checkpoints/model_fold_4.bin', map_location=torch.device('cpu')))
    MODEL.to(device)
    app.run(debug=True)

import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import pandas as pd
import pretrainedmodels


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



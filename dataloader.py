"""
__author__: Abhishek Thakur
"""
import cv2
import torch

import numpy as np

from PIL import Image
from PIL import ImageFile

try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    def __init__(
        self,
        image_paths,
        targets,
        resize,
        augmentations=None,
        backend="pil",
        channel_first=True,
    ):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param resize: tuple or None
        :param augmentations: albumentations augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.backend = backend
        self.channel_first = channel_first

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        targets = self.targets[item]
        if self.backend == "pil":
            image = Image.open(self.image_paths[item])
            if self.resize is not None:
                image = image.resize(
                    (self.resize[1], self.resize[0]), resample=Image.BILINEAR
                )
            image = np.array(image)
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
        elif self.backend == "cv2":
            image = cv2.imread(self.image_paths[item])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.resize is not None:
                image = cv2.resize(
                    image,
                    (self.resize[1], self.resize[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
            image = augmented["image"]
        else:
            raise Exception("Backend not implemented")
        if self.channel_first:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.tensor(image),
            "targets": torch.tensor(targets),
        }


class ClassificationDataLoader:
    def __init__(
        self,
        image_paths,
        targets,
        resize,
        augmentations=None,
        backend="pil",
        channel_first=True,
    ):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param resize: tuple or None
        :param augmentations: albumentations augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.backend = backend
        self.channel_first = channel_first
        self.dataset = ClassificationDataset(
            image_paths=self.image_paths,
            targets=self.targets,
            resize=self.resize,
            augmentations=self.augmentations,
            backend=self.backend,
            channel_first=self.channel_first,
        )

    def fetch(
        self,
        batch_size,
        num_workers,
        drop_last=False,
        shuffle=True,
        tpu=False,
        sampler=None,
    ):
        """
        :param batch_size: batch size
        :param num_workers: number of processes to use
        :param drop_last: drop the last batch?
        :param shuffle: True/False
        :param tpu: True/False, to use tpu or not
        """
        if tpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle,
            )

        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        return data_loader

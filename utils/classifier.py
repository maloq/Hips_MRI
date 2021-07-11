from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

class MRDataset(data.Dataset):

    def __init__(self, task, plane, train, drop_researh_tags=None, researh_type=None, resize=True, dim=(256, 256),
                 crop=True, transform=True, best_slice=True, weights=None, ):
        super().__init__()
        if drop_researh_tags is None:
            drop_researh_tags = ['DIRTY', 'TRIM', 'STIR']

        PPDatset = PreprocessedDatasetPart1()
        data_list = PPDatset.make_in_memory_dataset(task=task, projection=plane, researh_types=researh_type,
                                                    drop_types=drop_researh_tags, crop=True, best_slice=True,
                                                    resize=True, dim=(256, 256))
        self.task = task
        self.plane = plane
        self.train = train

        data_list_train, data_list_test = train_test_split(data_list, test_size=0.3, shuffle=False)
        self.length = len(data_list)
        self.transform = transform
        if self.train:
            self.data = data_list_train
        else:
            self.transform = False
            self.data = data_list_test

        if self.transform:
            self.transforms = A.ReplayCompose([
                A.Rotate(limit=5),
                A.HorizontalFlip()
            ])

            self.transforms_random = A.Compose([
                A.CLAHE(),
                A.GaussNoise(var_limit=(0.0, 30.0)),
                A.MedianBlur(blur_limit=3),
                A.ElasticTransform(alpha=1, sigma=50),
                A.RandomGamma(gamma_limit=(60, 140)),
                A.RandomFog(fog_coef_lower=0.0, fog_coef_upper=0.3)
            ])
            
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        array = self.data[index]['images']
        label = self.data[index]['label']
        if label == 1:
            label = torch.FloatTensor([[0, 1]])
        elif label == 0:
            label = torch.FloatTensor([[1, 0]])

        if self.transform:
            array = self.make_transforms(array, self.transforms)
            array = self.make_random_transforms(array, self.transforms_random)
        else:
            array = torch.from_numpy(array)
        return array, label, 1

    def make_transforms(self, array, transforms):
        random.seed(42)
        i = 1
        data = transforms(image=np.moveaxis(array[0], 0, -1))
        transformed = data['image']
        transformed = np.moveaxis(transformed, -1, 0)
        # transformed = np.moveaxis(transformed, 1, 0)
        array[0] = transformed
        replay = data['replay']
        for image in array[1:]:
            transformed = A.ReplayCompose.replay(replay, image=np.moveaxis(image, 0, -1))['image']
            # transformed = np.moveaxis(transformed, 1, 0)
            array[i] = np.moveaxis(transformed, -1, 0)
            i += 1
        return array

    def make_random_transforms(self, array, random_transforms):

        for i, image in enumerate(array):
            transformed = random_transforms(image=np.moveaxis(image, 0, -1))
            transformed_image = transformed["image"]
            array[i] = np.moveaxis(transformed_image, -1, 0)
        return array
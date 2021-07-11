import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine, distortion_transforms

from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dataloader import MRDataset
import model

from sklearn import metrics

train_dataset = MRDataset('cartialge lesion', 'cor',drop_researh_tags =['DIRTY', 'TRIM', 'STIR'],  researh_type = ['T1', 'T2', 'PD'], train=False, transform=None)

#train_loader = torch.utils.data.DataLoader(
#        train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

for i in range(len(train_dataset)):
    array, label, path = train_dataset[i]

    # print(i, array.size(), label, path)

    if i == 100:
        break
    # if torch.randint(0, 2, (1,)).item() == 1:
    #   random_label = [0, 1]
    # else:
    #   random_label = [1, 0]

    print(array.shape, label, path)

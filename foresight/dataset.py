
# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
from collections import namedtuple
from .imagenet16 import *
import torch
import json

def convert_param(original_lists):
      ctype, value = original_lists[0], original_lists[1]
      is_list = isinstance(value, list)
      if not is_list: value = [value]
      outs = []
      for x in value:
        if ctype == 'int':
          x = int(x)
        elif ctype == 'str':
          x = str(x)
        elif ctype == 'bool':
          x = bool(int(x))
        elif ctype == 'float':
          x = float(x)
        elif ctype == 'none':
          if x.lower() != 'none':
            raise ValueError('For the none type, the value must be none instead of {:}'.format(x))
          x = None
        else:
          raise TypeError('Does not know this type : {:}'.format(ctype))
        outs.append(x)
      if not is_list: outs = outs[0]
      return outs
def get_cifar_dataloaders(train_batch_size, test_batch_size, dataset, num_workers, resize=None, datadir='_dataset',if_split = False,pad = True,meanstd = None):

    if 'ImageNet16' in dataset:
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
        size, pad = 16, 2
    elif 'cifar' in dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size, pad = 32, 4
    elif 'svhn' in dataset:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        size, pad = 32, 0
    elif dataset == 'ImageNet1k':
        from .h5py_dataset import H5Dataset
        size,pad = 224,2
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        #resize = 256
    if meanstd is not None:
        mean,std = meanstd
    if resize is None:
        resize = size
    if pad is False:
        pad = 0
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(size, padding=pad),
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

    if dataset == 'cifar10':
        train_dataset = CIFAR10(datadir, True, train_transform, download=True)
        test_dataset = CIFAR10(datadir, False, test_transform, download=True)
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(datadir, True, train_transform, download=True)
        test_dataset = CIFAR100(datadir, False, test_transform, download=True)
    elif dataset == 'svhn':
        train_dataset = SVHN(datadir, split='train', transform=train_transform, download=True)
        test_dataset = SVHN(datadir, split='test', transform=test_transform, download=True)
    elif dataset == 'ImageNet16-120':
        train_dataset = ImageNet16(os.path.join(datadir, 'ImageNet16'), True , train_transform, 120)
        test_dataset  = ImageNet16(os.path.join(datadir, 'ImageNet16'), False, test_transform , 120)
    elif dataset == 'ImageNet1k':
        train_dataset = H5Dataset(os.path.join(datadir, 'imagenet-train-256.h5'), transform=train_transform)
        test_dataset  = H5Dataset(os.path.join(datadir, 'imagenet-val-256.h5'),   transform=test_transform)
            
    else:
        raise ValueError('There are no more cifars or imagenets.')
    if if_split:
        with open(os.path.join(datadir,'cifar-split.txt'), 'r') as f:
            data = json.load(f)
            content = { k: convert_param(v) for k,v in data.items()}
            Arguments = namedtuple('Configure', ' '.join(content.keys()))
            content = Arguments(**content)
        cifar_split = content
        train_split, valid_split = cifar_split.train, cifar_split.valid
        train_loader = DataLoader(
            train_dataset,
            train_batch_size,
            shuffle=False,
            num_workers=num_workers,sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
            pin_memory=True)
    else:
        train_loader = DataLoader(
            train_dataset,
            train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, test_loader


def get_mnist_dataloaders(train_batch_size, val_batch_size, num_workers):

    data_transform = Compose([transforms.ToTensor()])

    # Normalise? transforms.Normalize((0.1307,), (0.3081,))

    train_dataset = MNIST("_dataset", True, data_transform, download=True)
    test_dataset = MNIST("_dataset", False, data_transform, download=True)

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, test_loader


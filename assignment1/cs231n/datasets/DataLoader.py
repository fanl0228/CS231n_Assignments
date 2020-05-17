"""
This file is a dataloader
use torch

Author: Long Fan
Date: 2020-5-17
github: https://github.com/fanl0228

version: 1.0

use torchvision.datasets ref-link:
https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-datasets/

"""
import os
import torch.utils.data as data
import pickle
from PIL import Image
import numpy as np


class CIFAR10DataLoader(data.Dataset):

    def __init__(self, root, train, transform, target_transform):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []

        if self.train:
            choose_list = self.train_list
        else:
            choose_list = self.test_list

        for file_name in choose_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])

                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    train_list = ["data_batch_1", "data_batch_2", "data_batch_3",
                  "data_batch_4", "data_batch_5"]
    test_list = ["test_batch"]





















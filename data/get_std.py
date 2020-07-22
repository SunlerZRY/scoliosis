import os
import random

import torch

from data.DegreesData import DegreesData


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=32)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


train_data_path = [
    "/data/gukedata/train_data/0-10",
    "/data/gukedata/train_data/11-15",
    "/data/gukedata/train_data/16-20",
    "/data/gukedata/train_data/21-25",
    "/data/gukedata/train_data/26-45",
    "/data/gukedata/train_data/46-"
]
train_dataset = DegreesData(train_data_path, sample=False)
mean, std = get_mean_and_std(train_dataset)
print("mean:", mean)
print("std:", std)


# mean: tensor([0.4771, 0.4769, 0.4355])
# std: tensor([0.2189, 0.1199, 0.1717])


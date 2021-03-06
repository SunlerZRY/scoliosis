import os
import random

from PIL import Image, ImageFilter
from RandAugment import RandAugment
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import deepdish as dd

from .CutPicture import cutPicture


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class DegreesData(Dataset):
    def __init__(self, labels_path, image_size, istraining=False, sample=True):
        normalize = transforms.Normalize(mean=[0.4771, 0.4769, 0.4355],
                                         std=[0.2189, 0.1199, 0.1717])
        self.istraining = istraining
        self.images = dd.io.load(labels_path)

        if istraining:
            self.transform = transforms.Compose([
                transforms.ColorJitter(
                    64.0 / 255, 0.75, 0.25, 0.04),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomApply(
                    [GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(),
                # transforms.ToPILImage()
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                normalize,
            ])

        print(len(self.images))

    def sample_filter(self):
        samples = []
        for i in self.images:
            if i['degree'] <= 50:
                samples.append(i)
        return samples

    def smooth_label(self, label, threshold=30):
        if label > threshold:
            # TODO: make label smooth
            label = threshold+(label-threshold)*

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]

        img = Image.open(file['img_path'])
        cropbox = file['bbox']
        label = float(file['degree'])/3.0
        second_label = float(file['second_degree'])/3.0

        try:
            img = img.crop(cropbox)  # 后面可以追加更复杂的裁剪算法
            img = self.transform(img)

        except IOError:
            print(file)
            raise IOError("File is error, ", file)
        # for k in self.group_file_list.keys():
        #     if file in self.group_file_list[k]:
        #         label = k

        return img, torch.Tensor([label, second_label])


# listDirs = ['/home/hdc/yhf/Guke2/0-10', '/home/hdc/yhf/Guke2/11-15',
#             '/home/hdc/yhf/Guke2/16-20', '/home/hdc/yhf/Guke2/21-25',
#             '/home/hdc/yhf/Guke2/26-45', '/home/hdc/yhf/Guke2/46-']

# train_dataset = DegreesData(class_dirs=listDirs)

# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)

# print(len(train_loader))
# print("==============")

# for i_batch, batch_data in enumerate(train_loader):
#     print(i_batch)
#     image, label = batch_data
#     print(image.shape, label)

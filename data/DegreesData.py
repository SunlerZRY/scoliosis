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
    def __init__(self, class_dirs, image_size, istraining=False, sample=True):
        normalize = transforms.Normalize(mean=[0.4771, 0.4769, 0.4355],
                                         std=[0.2189, 0.1199, 0.1717])
        self.istraining = istraining
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
            self.group_file_list, self.images, self.labels = self.get_file_list(
                class_dirs, sample)

        else:
            self.transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                normalize,
            ])
            self.images = dd.io.load(
                '/data/gukedata/valid_data/labels_list.h5')
        print(len(self.images))

    def get_file_list(self, class_dirs, sample):
        group_file_list = {
            "0": [],
            "1": [],
            # "2": [],
            # "2": [],
            # "3": [],
            # "4": []
        }

        for class_dir in class_dirs:
            if "0-10" in class_dir:
                files = os.listdir(class_dir)
                for file in files:
                    if '.jpg' in file or '.JPG' in file:
                        group_file_list["0"].append(
                            os.path.join(class_dir, file))
            elif "11-15" in class_dir:
                files = os.listdir(class_dir)
                for file in files:
                    if '.jpg' in file or '.JPG' in file:
                        group_file_list["0"].append(
                            os.path.join(class_dir, file))
            elif "16-20" in class_dir:
                files = os.listdir(class_dir)
                for file in files:
                    if '.jpg' in file or '.JPG' in file:
                        group_file_list["1"].append(
                            os.path.join(class_dir, file))
            elif "21-25" in class_dir:
                files = os.listdir(class_dir)
                for file in files:
                    if '.jpg' in file or '.JPG' in file:
                        group_file_list["1"].append(
                            os.path.join(class_dir, file))
            elif "26-45" in class_dir:
                files = os.listdir(class_dir)
                for file in files:
                    if '.jpg' in file or '.JPG' in file:
                        group_file_list["1"].append(
                            os.path.join(class_dir, file))
            else:
                files = os.listdir(class_dir)
                for file in files:
                    if '.jpg' in file or '.JPG' in file:
                        group_file_list["1"].append(
                            os.path.join(class_dir, file))

        sample_path = []
        labels = []
        if sample:
            sample_num = min([len(n) for k, n in group_file_list.items()])
            for k, p in group_file_list.items():
                sample_path.extend(random.sample(p, int(sample_num)))
                labels.extend([k]*int(sample_num))
        else:
            for k, p in group_file_list.items():
                sample_path.extend(p)
                labels.extend([k]*len(p))

        return group_file_list, sample_path, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]
        if self.istraining:
            xmlFile = file.split('.')[0] + '.xml'
            img, cropbox = cutPicture(file, xmlFile)
            label = self.labels[index]
        else:
            img = Image.open(file['img_path'])
            cropbox = file['bbox']
            label_degree = file['degree']

            if label_degree >= 15:
                label = 1
            # elif label_degree < 20 and label_degree > 10:
            #     label = 1
            else:
                label = 0

        try:
            img = img.crop(cropbox)  # 后面可以追加更复杂的裁剪算法
            img = self.transform(img)

        except IOError:
            print(file)
            raise IOError("File is error, ", file)
        # for k in self.group_file_list.keys():
        #     if file in self.group_file_list[k]:
        #         label = k

        return img, int(label)


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

import os
import random
import bisect

from PIL import Image, ImageFilter
import xml.dom.minidom
import numpy as np
from RandAugment import RandAugment
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import deepdish as dd

# from .CutPicture import get_box

# TODO还没写完，需要修改模型的最后输出，分别预测度数和类别


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class DegreesData(Dataset):
    def __init__(self, labels_path, class_point, image_size, istraining=False, sample=True):
        normalize = transforms.Normalize(mean=[0.4771, 0.4769, 0.4355],
                                         std=[0.2189, 0.1199, 0.1717])
        self.istraining = istraining
        self.images = dd.io.load(labels_path)
        self.class_point = class_point
        group_list, self.samples = self.get_group_list_2(class_point, sample)

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
            group_file_list, samples = self.get_file_list_1(class_dirs,
                                                            sample)
            self.samples.extend(samples)

        else:
            self.transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                normalize,
            ])

        print(len(self.samples))

    def get_group_list_2(self, class_point, sample):
        """[summary]

        Parameters
        ----------
        class_point : list
            like [10, 15, 20]
        sample : bool
            if use sampling to balance the group

        Returns
        -------
        [type]
            [description]
        """
        group_file_list = {
        }
        class_num = len(class_point)+1
        for i in range(class_num):
            group_file_list[str(i)] = []

        for img_dir in self.images:
            label_degree = img_dir['degree']
            label = bisect.bisect_left(class_point, label_degree)
            # print(label_degree, label)
            group_file_list[str(label)].append(img_dir)

        samples = []
        labels = []
        tag = False
        if sample:
            sample_num = min([len(n) for k, n in group_file_list.items()])
            # if k == class_num-1:
            # sample_num = int(sample_num * 3.0 / 4.0)
            #     tag = True
            for k, p in group_file_list.items():
                # if k == str(class_num - 1):
                #     sample_num = int(sample_num * 4.0 / 3.0) - 1
                    # print('len', str(class_num - 1), sample_num)
                samples.extend(random.sample(p, int(sample_num)))
        else:
            for k, p in group_file_list.items():
                samples.extend(p)

        return group_file_list, samples

    def get_file_list_1(self, class_dirs, sample):
        group_file_list = {'0': [],
                           '1': [],
                           '2': [],
                           '3': [],
                           '4': []}

        for class_dir in class_dirs:
            if "0-10" in class_dir:
                files = os.listdir(class_dir)
                for file in files:
                    if '.jpg' in file or '.JPG' in file:
                        path = os.path.join(class_dir, file)
                        xmlFile = path.split('.')[0] + '.xml'
                        cropbox = get_box(xmlFile)
                        group_file_list["0"].append(
                            {'img_path': path, 'bbox': cropbox, 'degree': 8, 'type': '/'})
            elif "11-15" in class_dir:
                files = os.listdir(class_dir)
                for file in files:
                    if '.jpg' in file or '.JPG' in file:
                        path = os.path.join(class_dir, file)
                        xmlFile = path.split('.')[0] + '.xml'
                        cropbox = get_box(xmlFile)
                        group_file_list["1"].append(
                            {'img_path': path, 'bbox': cropbox, 'degree': 12, 'type': 'o'})
            elif "16-20" in class_dir:
                files = os.listdir(class_dir)
                for file in files:
                    if '.jpg' in file or '.JPG' in file:
                        path = os.path.join(class_dir, file)
                        xmlFile = path.split('.')[0] + '.xml'
                        cropbox = get_box(xmlFile)
                        group_file_list["2"].append(
                            {'img_path': path, 'bbox': cropbox, 'degree': 17, 'type': 'o'})
            elif "21-25" in class_dir:
                files = os.listdir(class_dir)
                for file in files:
                    if '.jpg' in file or '.JPG' in file:
                        path = os.path.join(class_dir, file)
                        xmlFile = path.split('.')[0] + '.xml'
                        cropbox = get_box(xmlFile)
                        group_file_list["3"].append(
                            {'img_path': path, 'bbox': cropbox, 'degree': 22, 'type': 'o'})
            elif "26-45" in class_dir:
                files = os.listdir(class_dir)
                for file in files:
                    if '.jpg' in file or '.JPG' in file:
                        path = os.path.join(class_dir, file)
                        xmlFile = path.split('.')[0] + '.xml'
                        cropbox = get_box(xmlFile)
                        group_file_list["4"].append(
                            {'img_path': path, 'bbox': cropbox, 'degree': 26, 'type': 'o'})
            # else:
            #     files = os.listdir(class_dir)
            #     for file in files:
            #         if '.jpg' in file or '.JPG' in file:
            #             path = os.path.join(class_dir, file)
            #             xmlFile = path.split('.')[0] + '.xml'
            #             cropbox = get_box(xmlFile)
            #             group_file_list["4"].append(
            #                 {'img_path': path, 'bbox': cropbox, 'degree': 46})
        # print("group_file_list:", len(
        #     group_file_list['0']), len(group_file_list['1']), len(group_file_list['2']))

        samples = []
        labels = []
        tag = False
        if sample:
            sample_num = min([len(n) for k, n in group_file_list.items()])
            # print(sample_num)
            # sample_num = int((sample_num * 3.0) / 4.0)
            # print('3/4', sample_num)
            for k, p in group_file_list.items():
                # if k == '2':
                #     sample_num = int((sample_num * 4.0) / 3.0) - 1
                    # print('len of 2', sample_num)
                samples.extend(random.sample(p, int(sample_num)))
        else:
            for k, p in group_file_list.items():
                samples.extend(p)
        # print(len(samples))
        return group_file_list, samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file = self.samples[index]

        img = Image.open(file['img_path'])
        cropbox = file['bbox']
        label_degree = file['degree']

        try:
            img = img.crop(cropbox)  # 后面可以追加更复杂的裁剪算法
            img = self.transform(img)

            label = bisect.bisect_right(self.class_point, label_degree)

        except IOError:
            print(file)
            raise IOError("File is error, ", file)

        labels = [0]*(len(self.class_point))
        for i in range(label):
            labels[i] = 1

        mask = True
        types = 0
        if file['type'] is 'o':
            mask = False
        if file['type'] is '/' and label_degree > 9:
            mask = False
        if file['type'] is '四弯' or '半椎体' or '上胸弯':
            mask = False
        if file['type'] is '胸弯':
            types = 0
        elif file['type'] is '腰弯':
            types = 1
        elif file['type'] is '胸腰弯':
            types = 2
        elif file['type'] is '双弯':
            types = 3
        elif file['type'] is '三弯':
            types = 4
        elif mask == True:
            print("Error!!!!! file['type']:", file['type'])

        return img, torch.Tensor(labels), label, label_degree, types, mask


def get_box(xmlFile):
    DomTree = xml.dom.minidom.parse(xmlFile)
    annotation = DomTree.documentElement
    objectlist = annotation.getElementsByTagName('object')
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data
        bndbox = objects.getElementsByTagName('bndbox')
        cropboxes = []
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
            w = x2 - x1
            h = y2 - y1
            obj = np.array([x1, y1, x2, y2])
            shift = np.array([[1, 1, 1, 1]])
            XYmatrix = np.tile(obj, (1, 1))
            cropboxes = XYmatrix * shift

            for cropbox in cropboxes:
                # img = img.crop(cropbox)
                # img = img.resize((scale, scale), Image.NEAREST)  # 长宽都为scale
                return cropbox

# listDirs = ['/home/hdc/yhf/Guke2/0-10', '/home/hdc/yhf/Guke2/11-15',
#             '/home/hdc/yhf/Guke2/16-20', '/home/hdc/yhf/Guke2/21-25',
#             '/home/hdc/yhf/Guke2/26-45', '/home/hdc/yhf/Guke2/46-']


# train_dataset = DegreesData(
#     "/data/gukedata/org_data/train_labels_list.h5", [10, 20], 255, istraining=True, sample=True)

# train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# print(len(train_loader))
# print("==============")
# n0 = 0
# n1 = 0
# n2 = 0
# for i_batch, batch_data in enumerate(train_loader):
#     print(i_batch)

#     image, label, mask = batch_data
#     n0 += int((label == 0).sum())
#     n1 += int((label == 1).sum())
#     n2 += int((label == 2).sum())
#     print(image.shape, label, n0, n1, n2)

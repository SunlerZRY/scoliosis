import os
import sys


import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch.nn import functional as F
from apex import amp
import pandas as pd

sys.path.append('/home/hdc/guke/')  # noqa

from inference.inference_bbox import init_bbox_detector, inference_img
from data.CutPicture import cutPicture


def load_checkpoint(path, net):
    print("Use ckpt: ", path)
    # checkpoint = torch.load(args.ckpt)
    net = amp.initialize(net,
                         opt_level="O1",
                         )
    checkpoint = torch.load(
        path, map_location=lambda storage, loc: storage.cuda(0))
    pretrained_dict = checkpoint['state_dict']
    net.load_state_dict(pretrained_dict)
    epoch = checkpoint['epoch']
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # amp.load_state_dict(checkpoint['amp'])
    return net, epoch


config_file = '/home/hdc/mmdetection/configs/retinanet/retinanet_r50_fpn_f16_finetun_m.py'
checkpoint_file_bbox = '/home/hdc/mmdetection/work_dirs/retinanet_r50_fpn_f16_finetun_m/latest.pth'
checkpoint_file_class = '/data/gukedata/ckpt/efficientnet_nc_5_aug_adam_mixup_3/best.ckpt'

data_path = '/data/gukedata/valid_data/valid_set/'

bbox_model = init_bbox_detector(config_file, checkpoint_file_bbox)
# result = inference_img(bbox_model, img)
# print(result)

class_model = EfficientNet.from_name('efficientnet-b7', {'num_classes': 5})
class_model = class_model.to('cuda:0')
class_model, epoch = load_checkpoint(checkpoint_file_class, class_model)


imgs_list = []
for file in os.listdir(data_path):
    if '.jpg' in file or '.JPG' in file or '.JPEG' in file:
        path = os.path.join(data_path, file)
        imgs_list.append(path)

print("Num of samples:", len(imgs_list))

normalize = transforms.Normalize(mean=[0.4771, 0.4769, 0.4355],
                                 std=[0.2189, 0.1199, 0.1717])
transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    normalize,
])

labels = [
    "0~10",
    "11~15",
    "16~20",
    "21~25",
    "26+"
]
result = []
file_list = []
with torch.no_grad():
    for idx, img_path in enumerate(imgs_list):
        print(img_path)
        bbox_result = inference_img(bbox_model, img_path)
        img = Image.open(img_path)
        print(bbox_result)
        img = img.crop(bbox_result)
        img = transform(img).unsqueeze(0).cuda()
        class_result = class_model(img)
        probs = F.softmax(class_result, dim=1)
        _, predicts = torch.max(probs, 1)
        class_id = predicts.detach().cpu().numpy()[0]
        print(idx, labels[class_id],
              probs.detach().cpu().numpy())
        result.append(labels[class_id])
        file_list.append(img_path.split('/')[-1])

dataframe = pd.DataFrame({'img_name': file_list, 'pred_result': result})
dataframe.to_csv("result.csv", index=False, sep=',')

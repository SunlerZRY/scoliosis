# -*- coding:utf-8 _*-
import os
import sys

import deepdish as dd
import pandas as pd
from PIL import Image

sys.path.append('/home/hdc/guke/')  # noqa

from inference.inference_bbox import init_bbox_detector, inference_img
from data.CutPicture import cutPicture


def get_degree(str):
    if isinstance(str, float):
        degree = str
    elif isinstance(str, int):
        degree = str
    elif (str)[-1] == '/':
        degree = 0
    elif (str)[-1] == '°':
        degree = float((str)[:-1])
    elif len(str) < 5:
        degree = float(str)
    else:
        degree = -1
    return degree


config_file = '/home/hdc/mmdetection/configs/retinanet/retinanet_r50_fpn_f16_finetun_m.py'
checkpoint_file_bbox = '/home/hdc/mmdetection/work_dirs/retinanet_r50_fpn_f16_finetun_m/latest.pth'


label_path = '/data/gukedata/org_data/new/train_labels.xlsx'
data_path = '/data/gukedata/org_data/new/final/'
save_label = '/data/gukedata/org_data/'
df = pd.read_excel(label_path)
bbox_model = init_bbox_detector(config_file, checkpoint_file_bbox)

valid_set = dd.io.load('/data/gukedata/valid_data/valid_labels_list.h5')
valid_set_id = []
labels_list = []
index_list = []
for i in valid_set:
    valid_set_id.append(str(i["id"]).strip())
valid_set_id = set(valid_set_id)
print(valid_set_id)

for index, row in df.iterrows():
    img_name = row['外观片']
    if 'final' in img_name:
        img_name = row['外观片'].split('final')[-1][1:].strip()
        # print(img_name)
    # 排除验证集中的样本
    if str(row['病人ID']).strip() in valid_set_id:
        print(row['病人ID'], img_name, "in valid set")
        index_list.append(index)
        continue
    # else:

    #     if 'DSCN4546.JPG' != img_name:
    #         continue

    # 排除异常数据
    if row['主弯度数'] == '/' and row['主弯顶椎'] != '/':
        print(img_name, "is error")
        continue
    img_path = os.path.join(data_path, img_name).strip()

    if not os.path.isfile(img_path):
        img_path = img_path.split('.')[0] + '.jpg'

    try:
        result = inference_img(bbox_model, img_path)
    except:
        print('No such file:', img_path)
        continue

    # print(index, row['外观片'], row['主弯度数'])

    degree = get_degree(row['主弯度数'])
    second_degree = get_degree(row['次弯度数'])
    # 排除异常数据
    if degree == -1 or second_degree == -1:
        continue

    labels_list.append({"id": str(row['病人ID']).strip(),
                        "img_path": img_path,
                        "bbox": result,
                        "degree": degree,
                        "tip_cone": row['主弯顶椎'],
                        "second_degree": second_degree,
                        "second_tip_cone": row['次弯顶椎'],
                        "type": row['类型']})
    index_list.append(index)


# 将异常数据保存
df.drop(index=index_list, inplace=True)
df.to_excel('error_output.xls', sheet_name='error')

dd.io.save(save_label + 'train_labels_list.h5', labels_list)

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


label_path = '/data/gukedata/valid_data/valid_labels.xlsx'
data_path = '/data/gukedata/valid_data/valid_set/'
save_label = '/data/gukedata/valid_data/'
df = pd.read_excel(label_path)
bbox_model = init_bbox_detector(config_file, checkpoint_file_bbox)


labels_list = []


for index, row in df.iterrows():
    img_name = row['外观片']

    # 排除异常数据
    if row['主弯度数'] == '/' and row['主弯顶椎'] != '/':
        print(img_name, "is error")
        continue
    img_path = data_path + img_name

    if os.path.isfile(img_path):
        result = inference_img(bbox_model, img_path)
        print(index, row['外观片'], row['主弯度数'])

        degree = get_degree(row['主弯度数'])
        second_degree = get_degree(row['次弯度数'])
        # 排除异常数据
        if degree == -1 or second_degree == -1:
            continue

        labels_list.append({"img_path": img_path,
                            "bbox": result,
                            "degree": (row['主弯度数']),
                            "tip_cone": row['主弯顶椎'],
                            "second_degree": row['次弯度数'],
                            "second_tip_cone": row['次弯顶椎'],
                            "type": row['类型']})
    else:
        print('No such file:', img_path)
dd.io.save(save_label + 'valid_labels_list.h5', labels_list)

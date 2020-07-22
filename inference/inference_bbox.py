import sys
import os

import numpy as np
import mmcv

sys.path.append('/home/hdc/mmdetection')  # noqa

from mmdet.apis import init_detector, inference_detector, show_result_pyplot  # noqa


def inference_dir(model, dir_path,):
    imgs_path = []
    results = []
    for f in os.listdir(dir_path):
        #         print(f)
        if f.split('.')[-1] == 'JPG' or f.split('.')[-1] == 'jpg':
            imgs_path.append(os.path.join(dir_path, f))
    print("num of imgs:", len(imgs_path))
    for k, img in enumerate(imgs_path):
        if k > 20:
            break
        result = inference_detector(model, img)
        # print(k, img, result)
        # show_result_pyplot(model, img, result, score_thr=0.5)
        results.append(result[0][0, :4])

    return imgs_path, results


def inference_img(model, path):

    result = inference_detector(model, path)
    # print(path, result)
    # show_result_pyplot(model, path, result, score_thr=0.5)

    return result[0][0, :4]


def init_bbox_detector(config_file, checkpoint_file):

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    return model


# config_file = '/home/hdc/mmdetection/configs/retinanet/retinanet_r50_fpn_f16_finetun_m.py'
# checkpoint_file = '/home/hdc/mmdetection/work_dirs/retinanet_r50_fpn_f16_finetun_m/latest.pth'
# img = '/data/gukedata/test_data/11-15/1023.jpg'

# model = init_bbox_detector(config_file, checkpoint_file)
# result = inference_img(model, img)
# print(result)

from __future__ import division
from PIL import Image
import xml.dom.minidom
import numpy as np


def cutPicture(file, xmlFile, scale=256):
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
            img = Image.open(file)
            for cropbox in cropboxes:
                # img = img.crop(cropbox)
                # img = img.resize((scale, scale), Image.NEAREST)  # 长宽都为scale
                return img, cropbox

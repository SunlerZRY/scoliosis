# 生成训练集和测试集目录

import random
import os
import shutil

origin_dirs = ["/data/gukedata/org_data/0-10", "/data/gukedata/org_data/11-15",
               "/data/gukedata/org_data/16-20", "/data/gukedata/org_data/21-25",
               "/data/gukedata/org_data/26-45", "/data/gukedata/org_data/46-"]

train_path = "/data/gukedata/train_data"
test_path = "/data/gukedata/test_data"

# origin_dirs = ["E:/gukedata/0-10", "E:/gukedata/11-15",
#                "E:/gukedata/16-20", "E:/gukedata/21-25",
#                "E:/gukedata/26-45", "E:/gukedata/46-"]
#
# train_path = "E:/data/train_data"
# test_path = "E:/data/test_data"

for origin_dir in origin_dirs:
    print("正在处理" + origin_dir)
    listOrigin = os.listdir(origin_dir)
    listPicture = []

    num = int(len(listOrigin) / 2 * 0.71)

    for i in range(len(listOrigin)):
        if '.jpg' in listOrigin[i] or '.JPG' in listOrigin[i]:
            listPicture.append(listOrigin[i])

    random.shuffle(listPicture)

    result = []

    print(len(listPicture))

    result.append(listPicture[0: num])
    result.append(listPicture[num:])

    print(len(result))

    for i in range(len(result)):
        if i == 0:
            for j in result[i]:
                name = j.split(".")[0]
                xmlFile = origin_dir + '/' + name + '.xml'
                file = origin_dir + '/' + j
                des_path = train_path + '/' + origin_dir.split("/")[-1]
                # print("复制到" + des_path)
                shutil.copy2(file, des_path)
                shutil.copy2(xmlFile, des_path)
        if i == 1:
            for j in result[i]:
                name = j.split(".")[0]
                xmlFile = origin_dir + '/' + name + '.xml'
                file = origin_dir + '/' + j
                des_path = test_path + '/' + origin_dir.split("/")[-1]
                # print("复制到" + des_path)
                shutil.copy2(file, des_path)
                shutil.copy2(xmlFile, des_path)

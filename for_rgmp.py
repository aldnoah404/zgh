import os
from PIL import Image

root_path = r'C:\Users\chenj\Desktop\曾国豪毕业资料\毕业设计试验程序及结果\dataset\Youtube-VOS\Youtube-VOS'
train_path = os.path.join(root_path, r'train\JPEGImages')
valid_path = os.path.join(root_path, r'valid\JPEGImages')
train_list = os.listdir(train_path)
train_list.sort(key=lambda x : (x.split('_')[0] ,int(x.split('_')[1])))
valid_list = os.listdir(valid_path)
valid_list.sort(key=lambda x : (x.split('_')[0] ,int(x.split('_')[1])))

train_txt = os.path.join(root_path, 'train.txt')
with open(train_txt, 'w') as f:
    for i in train_list:
        print(i, file=f)

valid_txt = os.path.join(root_path, 'valid.txt')
with open(valid_txt, 'w') as f:
    for i in valid_list:
        print(i, file=f)
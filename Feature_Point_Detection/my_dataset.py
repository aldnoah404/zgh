from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import numpy as np
import torchvision
from torchvision.transforms import functional as F
import math
import random


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for trans in self.transforms:
            image, target = trans(image, target)
        return image, target
    
class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        
        image = F.to_tensor(image).contiguous()
        return image, target
    
class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes,该方法应放在ToTensor后"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # height, width = image.shape[-2:]
            # image = image.flip(-1)  # 水平翻转图片
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转图片
            target[0] = 1.0 - target[0]  # 翻转对应坐标信息
        return image, target
    
class Normalize(object):
    """对图像标准化处理,该方法应放在ToTensor后"""
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, image, target):
        image = self.normalize(image)
        return image, target
    
class RandomRotation(object):

    def __init__(self, degrees):
        assert isinstance(degrees, tuple) or isinstance(degrees, list)
        assert len(degrees) == 2
        self.degrees = degrees

    def __call__(self, image, target):

        degree = random.randint(self.degrees[0], self.degrees[1])
        image = image.rotate(degree)

        center = (1.0/2, 1.0/2)
        # 将角度转换为弧度
        angle_radians = math.radians(degree)
        cos_theta = math.cos(angle_radians)
        sin_theta = math.sin(angle_radians)
        # 坐标旋转公式，这里需要将点坐标减去旋转中心，旋转后再加上
        new_x = (target[0] - center[0]) * cos_theta + (target[1] - center[1]) * sin_theta + center[0]
        new_y = -(target[0] - center[0]) * sin_theta + (target[1] - center[1]) * cos_theta + center[1]
        target[0] = new_x
        target[1] = new_y

        return image, target

data_transform = Compose([RandomRotation((-90,90)),
                          RandomHorizontalFlip(),
                          ToTensor(),
                          Normalize([0.07, 0.07, 0.07], [0.26, 0.26, 0.26])])

class MyDataSet(Dataset):

    def __init__(self, root, transforms=data_transform, train=True):

        if train:
            txt = os.path.join(root, 'train.txt')
        else:
            txt = os.path.join(root, 'val.txt')

        with open(txt, 'r') as f:
            self.sample_list = f.readlines()
        self.sample_list = [x.strip() for x in self.sample_list]

        self.sample_root = os.path.join(root, 'dataset')

        self.transforms = transforms

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.sample_root, self.sample_list[idx]+'.png')
        image = Image.open(img_path).convert('RGB')

        point_path = os.path.join(self.sample_root, self.sample_list[idx]+'.txt')
        point = np.loadtxt(point_path)

        # 将坐标值转换成相对值0-1之间
        h, w = image.size
        point /= np.array([h, w])

        # convert everything into a torch.Tensor
        point = torch.as_tensor(point, dtype=torch.float32)

        if self.transforms is not None:
            image, point  = self.transforms(image, point)

        return image, point

    @staticmethod
    def collate_fn(batch):
        images, targets = tuple(zip(*batch))
        # images = torch.stack(images, dim=0)
        #
        # boxes = []
        # labels = []
        # img_id = []
        # for t in targets:
        #     boxes.append(t['boxes'])
        #     labels.append(t['labels'])
        #     img_id.append(t["image_id"])
        # targets = {"boxes": torch.stack(boxes, dim=0),
        #            "labels": torch.stack(labels, dim=0),
        #            "image_id": torch.as_tensor(img_id)}

        return images, targets
from __future__ import print_function, division
import os
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
import random
import numpy as np
import json


# Weld Seam Segmentation
class WSSegmentation(Dataset):
    def __init__(self, ws_root, train=True, txt_name: str = "train.txt"):
        super(WSSegmentation, self).__init__()
        root = ws_root
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'mask')

        txt_path = os.path.join(root, txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

        if train:
            self.input_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((288,288)),
                torchvision.transforms.RandomRotation((-10,10)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.3], std=[0.32]),
            ])
            self.label_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((288,288)),
                torchvision.transforms.RandomRotation((-10,10)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ])
        else:
            self.input_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((288,288)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.3], std=[0.32]),
            ])
            self.label_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((288,288)),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ])


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # img = Image.open(self.images[index]).convert('RGB')
        img = Image.open(self.images[index]).convert('L')
        target = Image.open(self.masks[index]).convert('L')

        seed=np.random.randint(0,2**32) # make a seed with numpy generator 

        # apply this seed to img tranfsorms
        random.seed(seed) 
        torch.manual_seed(seed)
        img = self.input_transforms(img)

        # apply this seed to target/label tranfsorms  
        random.seed(seed) 
        torch.manual_seed(seed)
        target = self.label_transforms(target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        # batched_targets = cat_list(targets, fill_value=255)
        batched_targets = cat_list(targets, fill_value=0)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))   # 计算h, w的最大值，返回tuple(max_h, max_w)
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)   # batched_imgs是shape为batch_shape，元素值全为fill_value的tensor
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# Video Weld Seam Segmentation
class VideoSegmentation(Dataset):
    def __init__(self, root='C:\Users\chenj\Desktop\曾国豪毕业资料\毕业设计试验程序及结果\dataset\Youtube-VOS\Youtube-VOS',
                            train=True):
        super(VideoSegmentation, self).__init__()

        split = 'train' if train else 'valid'
        self.root = root  # 数据集根目录
        self.imgdir = os.path.join(root, split, 'JPEGImages')  # 图像根目录
        self.annodir = os.path.join(root, split, 'Annotations')  # 注释根目录

        with open(os.path.join(root, split, 'meta.json'), 'r') as f:
            meta = json.load(f)   # 数据集元文件

        self.info = meta['videos']
        self.videos = list(self.info.keys()) # 视频片段的名字列表

        if train:
            self.input_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((288,288)),
                torchvision.transforms.RandomRotation((-10,10)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.3], std=[0.32]),
            ])
            self.label_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((288,288)),
                torchvision.transforms.RandomRotation((-10,10)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ])
        else:
            self.input_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((288,288)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.3], std=[0.32]),
            ])
            self.label_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((288,288)),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ])


    def __getitem__(self, index):
        """
        获取一个视频片段的所有图像帧
        在一个iteration内只能按顺序前向计算所有视频帧, 不能并行计算
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        video_name = self.videos[index] # 视频片段的名字
        imgfolder = os.path.join(self.imgdir, video_name) # 视频片段对应的图像文件夹
        annofolder = os.path.join(self.annodir, video_name) # 视频片段对应的注释文件夹

        # 获取视频片段中的所有图像id并排序
        frames = [name[:5] for name in os.listdir(annofolder)]
        frames.sort()

        imgs = []
        masks = []
        for frame in frames:
            img = Image.open(os.path.join(imgfolder, frame+'.jpg')).convert('L')
            mask = Image.open(os.path.join(annofolder, frame+'.png')).convert('L')

            seed=np.random.randint(0,2**32) # make a seed with numpy generator 

            # apply this seed to img tranfsorms
            random.seed(seed) 
            torch.manual_seed(seed)
            img = self.input_transforms(img)

            # apply this seed to target/label tranfsorms  
            random.seed(seed) 
            torch.manual_seed(seed)
            mask = self.label_transforms(mask)

            imgs.append(img)
            masks.append(mask)

        return imgs, masks

    def __len__(self):
        # 返回数据集中视频片段的数量，每个视频片段都包含了10个图像帧
        return len(self.videos)
    

if __name__ == '__main__':

    import cv2
    from PIL import Image
    from matplotlib import pyplot as plt

    dataset = VideoSegmentation()
    print(len(dataset))

    imgs, masks = dataset[0]
    seq_len = len(imgs)

    # 需要先注释torchvision.transforms.ToTensor和torchvision.transforms.Normalize
    for i in range(seq_len):
        show = Image.new('RGB', (288+288+288, 288))

        img = imgs[i].convert('RGB')
        show.paste(img, (0, 0))

        mask = np.array(masks[i].convert('RGB'))
        mask[:, :, 1] = 0
        mask = Image.fromarray(mask)
        show.paste(mask, (288, 0))

        show2 = Image.blend(img, mask, 0.4)
        show.paste(show2, (288+288,0))

        plt.imshow(show)
        plt.show()

import os
import re
from glob import glob

import torch
import torchvision
import torch.nn.functional as F

import cv2
import numpy as np
from PIL import Image

from model import SSLSE
from loss import threshold_predictions_p


def get_metrics(prediction, target):
    """
    arges:
        prediction: np.array
        target: np.array
    
    Pixel Accuracy = TP / (TP + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    """

    assert prediction.shape == target.shape
    
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)

    TP = (prediction * target).sum()
    FP = prediction.sum() - TP
    FN = target.sum() - TP

    PA = (TP + target.shape[0] - TP - FP - FN) / target.shape[0]  # 这里的变量TP是激光条纹类别的TP。而背景的TP是整张图像的像素数减去激光条纹pred和target的并集，计算为(target.shape[0] - TP - FP - FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2. * Precision * Recall) / (Precision + Recall)

    # smooth = 1e-5  一般在loss函数中才引入
    smooth = 0.
    dice = (2. * TP + smooth) / (prediction.sum() + target.sum() + smooth)
    iou = (TP + smooth) / (prediction.sum() + target.sum() - TP + smooth)

    return PA, Precision, Recall, F1, iou, dice

def extract_numbers(s):
    s = os.path.basename(s)
    match = re.search(r'\d+', s)
    return int(match.group()) if match else 0


if __name__ == '__main__':

    img_root = '/zgh/dataset/WeldSeam/images'
    label_root = '/zgh/dataset/WeldSeam/mask'
    with open('/zgh/dataset/WeldSeam/val.txt', 'r') as f:
        file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    
    img_list = [os.path.join(img_root, x + ".jpg") for x in file_names]
    label_list = [os.path.join(label_root, x + ".png") for x in file_names]

    img_list =sorted(img_list, key=extract_numbers)
    label_list =sorted(label_list, key=extract_numbers)

    iou = np.zeros((len(img_list)))
    dice = np.zeros((len(img_list)))
    PA = np.zeros((len(img_list)))
    Precision = np.zeros((len(img_list)))
    Recall = np.zeros((len(img_list)))
    F1 = np.zeros((len(img_list)))

    data_transform = torchvision.transforms.Compose([
           torchvision.transforms.Resize((288,288)),
            torchvision.transforms.ToTensor(),
            # mean=(0.709, 0.381, 0.224), std=(0.127, 0.079, 0.043)
            # mean=[0.3], std=[0.32]
            torchvision.transforms.Normalize(mean=[0.3], std=[0.32])
        ])
    
    model = SSLSE(1, 1, 32)
    model.load_state_dict(torch.load('/zgh/Laser_Stripe_Extraction/exps/exp12/sslse_epoch_200_batchsize_8.pth'))
    model.eval().to(device='cuda')

    for i, (img, label) in enumerate(zip(img_list, label_list)):

        img = Image.open(img).convert('L')
        img = data_transform(img)
        img = img.unsqueeze(0).to(device='cuda')

        target = Image.open(label).convert('L')
        target = np.array(target) / 255.
        # target = cv2.resize(target, (288, 288))

        with torch.no_grad():
            pred = model(img)

        pred = F.sigmoid(pred).cpu()
        pred = pred[0][0].numpy()
        pred = cv2.resize(pred, (300, 300))
        pred = threshold_predictions_p(pred, 0.55)

        # print(pred.sum(), target.sum())
        # show = np.hstack([target, pred])
        # cv2.imshow('', show)
        # cv2.waitKey()
        # cv2.destroyWindow()

        PA[i], Precision[i], Recall[i], F1[i], iou[i], dice[i] = get_metrics(pred, target)

    print('Pixel Accuracy: {:.4f}'.format(PA.mean()))
    print('Precision: {:.4f}'.format(Precision.mean()))
    print('Recall: {:.4f}'.format(Recall.mean()))
    print('F1 score: {:.4f}'.format(F1.mean()))
    print('IoU: {:.4f}'.format(iou.mean()))
    print('Dice: {:.4f}'.format(dice.mean()))
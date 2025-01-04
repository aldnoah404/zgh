import torch
import torchvision
import numpy as np

import random
from PIL import ImageDraw
from matplotlib import pyplot as plt

from model import resnet34, resnet18
from my_dataset import MyDataSet

def main():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_set = MyDataSet('/zgh/datasets/key_point_detection', train=False)
    print(len(train_data_set))

    model = resnet18(num_classes=2)
    weights_path = "./resNet18.pth"
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    model.to(device)

    # index = random.randint(0, len(train_data_set)-1)
    indexs = random.sample(range(len(train_data_set)), 10)
    for index in indexs:
        img, target = train_data_set[index]
        img = img.unsqueeze(0)
        with torch.no_grad():
            pred = model(img.to(device)).cpu()
        img = img.squeeze(0)
        img = torchvision.transforms.ToPILImage()(img)
        target = target.squeeze(0).numpy()*288
        pred = pred.squeeze(0).numpy()*288
        draw = ImageDraw.Draw(img)
        # draw.point(target, fill=(255, 0, 0))
        draw.ellipse([target[0]-2, target[1]-2, target[0]+2, target[1]+2], fill=(255, 0, 0))
        draw.ellipse([pred[0]-2, pred[1]-2, pred[0]+2, pred[1]+2], fill=(0, 0, 255))
        print('Error: {:.2f}'.format(np.linalg.norm(target-pred, 2)))
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    main()

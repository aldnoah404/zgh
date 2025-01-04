import os
import re
import torch
import torch.nn.functional as F
import torchvision

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from model import SSLSE, VSLSE
from loss import threshold_predictions_p


img_root = '/zgh/data/sequence'  

re_digits = re.compile(r'(\d+)')
def embedded_numbers(s):
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces

img_list = os.listdir(img_root)
img_list = sorted(img_list, key=embedded_numbers)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VSLSE(1, 1, base_c=32)
model.load_state_dict(torch.load('/zgh/Laser_Stripe_Extraction/exps/exp13/vslse_epoch_200_batchsize_2.pth'))
model.to(device).eval()

input_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((288,288)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.3], std=[0.32])
                ])

for i, name in enumerate(img_list):

    if i == 0:
        model.init_hidden_state()

    img = Image.open(os.path.join(img_root, name)).convert('L')
    x = input_transforms(img)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)
    pred = F.sigmoid(pred)
    pred = pred[0][0].detach().cpu().numpy()

    pred = threshold_predictions_p(pred, 0.3)
    # pred *= 255
    pred = pred.astype(np.uint8)

    show = Image.new('L', (288+288+288, 288))

    img = img.resize((288, 288))
    show.paste(img, (0, 0))

    pred = Image.fromarray(pred)
    show.paste(pred, (288, 0))


        # show2 = Image.blend(img, mask, 0.4)
        # show.paste(show2, (288+288,0))

    plt.imshow(show)
    plt.show()
    

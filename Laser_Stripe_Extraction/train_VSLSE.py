from __future__ import print_function, division
import os
import numpy as np
from PIL import Image

from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F

import torch.nn
import torchvision
import matplotlib.pyplot as plt

from dataset import VideoSegmentation
from torch.utils.tensorboard import SummaryWriter


import shutil
import random
from model import VSLSE
from loss import calc_loss, dice_loss, threshold_predictions_p


#######################################################
# Checking if GPU is used
#######################################################

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
# Setting the basic paramters of the model
#######################################################

batch_size = 2
print('batch_size = ' + str(batch_size))

epoch = 200
print('epoch = ' + str(epoch))

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

valid_loss_min = np.Inf
lossT = []    # 训练损失
lossL = []    # 验证损失
lossL.append(np.inf)
lossT.append(np.inf)
n_iter = 1
i_valid = 0

pin_memory = False
if train_on_gpu:
    pin_memory = True

#######################################################
# Setting up the model
#######################################################

model_test = VSLSE(1, 1, base_c=32)

# 加载SSLSE模型的预训练权重
checkpoint_pre = torch.load('/zgh/Laser_Stripe_Extraction/exps/exp12/sslse_epoch_200_batchsize_8.pth')
checkpoint = model_test.state_dict()
checkpoint.update(checkpoint_pre)
model_test.load_state_dict(checkpoint)

model_test.to(device)

#######################################################
# Dataset and Dataloader
#######################################################

train_dataset = VideoSegmentation(train=True)
val_dataset = VideoSegmentation(train=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                batch_size=batch_size,
                                num_workers=0,
                                shuffle=True,
                                pin_memory=True)
valid_loader = torch.utils.data.DataLoader(val_dataset,
                                batch_size=1,
                                num_workers=0,
                                pin_memory=True)

#######################################################
# Using Adam as Optimizer
#######################################################

initial_lr = 1e-4
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr) # try SGD
# opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.9)

MAX_STEP = 64
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)
#scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

# 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
# scheduler = create_lr_scheduler(opt, len(train_loader), epoch, warmup=True)

#######################################################
# Creating a Folder for every data of the program
#######################################################

New_folder = './exp'

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    shutil.rmtree(New_folder)

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)

#######################################################
# gradient clipping
#######################################################

def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, torch.nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

#######################################################
# Training loop
#######################################################
writer1 = SummaryWriter(log_dir='./exp/log')
iter_id = 1
for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0
    scheduler.step(i)
    lr = scheduler.get_lr()

    #######################################################
    # Training Data
    #######################################################

    model_test.train()

    for X, Y in train_loader:

        opt.zero_grad()
        writer1.add_scalar('lr', scheduler.get_last_lr()[0], iter_id)

        for t in range(len(X)):

            if t == 0:
                # 初始化隐藏状态
                model_test.init_hidden_state()

            x, y = X[t].to(device), Y[t].to(device)
            y_pred = model_test(x)
            lossT = calc_loss(y_pred, y)     # bce+dice

            train_loss += lossT.item() * x.size(0)

        lossT.backward()
        grad_clipping(model_test, 1)  # 梯度裁剪
        opt.step()

        writer1.add_scalar('train_loss_iter', lossT, iter_id)
        iter_id += 1


    #######################################################
    # Validation Step
    #######################################################

    model_test.eval()
    torch.no_grad() #to increase the validation process uses less memory

    for X1, Y1 in valid_loader:

        for t in range(len(X1)):

            if t == 0:
                # 初始化隐藏状态
                model_test.init_hidden_state()

            x1, y1 = X1[t].to(device), Y1[t].to(device)

            y_pred1 = model_test(x1)
            lossL = calc_loss(y_pred1, y1)     # Dice_loss Used

            valid_loss += lossL.item() * x1.size(0)


    #######################################################
    # To write in Tensorboard
    #######################################################

    train_loss = train_loss / len(train_dataset)
    valid_loss = valid_loss / len(val_dataset)

    if (i+1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss))
        writer1.add_scalar('Loss/train loss', train_loss, n_iter)
        writer1.add_scalar('Loss/val loss', valid_loss, n_iter)
        writer1.add_scalars(main_tag='Loss/train_val loss',
                                                tag_scalar_dict={'train': train_loss,
                                                                                    'val': valid_loss},
                                                global_step=n_iter)
        #writer1.add_image('Pred', pred_tb[0]) #try to get output of shape 3


    #######################################################
    # save model
    #######################################################

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model_test.state_dict(),'./exp/' +
                                              'vslse_epoch_' + str(epoch)
                                              + '_batchsize_' + str(batch_size) + '.pth')
        valid_loss_min = valid_loss

        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print(i_valid)
            i_valid += 1

        # Early Stop
        # if i_valid == 5:
        #    break

#######################################################
# closing the tensorboard writer
#######################################################

writer1.close()

#######################################################
# checking if cuda is available
#######################################################

if torch.cuda.is_available():
    torch.cuda.empty_cache()

#######################################################
# Loading the model
#######################################################

model_test.load_state_dict(torch.load('./exp/' +
                        'vslse_epoch_' + str(epoch)
                        + '_batchsize_' + str(batch_size) + '.pth'))

model_test.eval()

#######################################################
# use a sample video clip from validation set 
#   for generating images 
#######################################################

idx = random.randint(0, len(val_dataset) - 1)
video_name = val_dataset.videos[idx]
print('Video name: ' + video_name)
imgs, masks = val_dataset[idx]
for t in range(len(imgs)):
    if t == 0:
        model_test.init_hidden_state()
    with torch.no_grad():
        pred = model_test(imgs[t].unsqueeze(0).to(device))
    pred = F.sigmoid(pred)
    pred = pred.detach().cpu().numpy()

    plt.imsave('./exp/' + video_name + str(t).zfill(4) + '.png', pred[0][0])

    pred = threshold_predictions_p(pred[0][0], 0.55)

    show = Image.fromarray(pred)
    plt.imshow(show)
    plt.show()

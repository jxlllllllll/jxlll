import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm, trange  # 进度条
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from Net import IENet
from dataset import ValueDataSet2
from psnr import psnr as psnr
from Loss import *
from dataset import *
net=IENet()
net=net.cuda()
net.load_state_dict(torch.load('./model_mist.pth'))

inputPathTrain="/home/liu/jxl/FiveK-Haze/1117mist/test_haze"
targetPathTrain="/home/liu/jxl/FiveK-Haze/1117mist/gt"


#resultPathTest = 'F:\\dehaze\\out\\'  # 测试结果图片路径

datasetTrain=ValueDataSet(inputPathTrain,targetPathTrain)
trainLoader=DataLoader(dataset=datasetTrain,batch_size=1)

net.eval()  # 指定网络模型测试状态

with torch.no_grad():
    #timeStart = time.time()
    for index, (x, y)in enumerate(tqdm(trainLoader, desc='Testing !!! ', file=sys.stdout), 0):
        input_, target_value = x.cuda(), y.cuda()
        output_value = net(input_)
        #print(output_test, resultPathTest + str(index + 1).zfill(3) + ".png")
        cur_psnr=psnr(output_value,target_value)
        #print(index)
       # print(cur_loss)
        psnr_val_rgb = []
        for output_value, target_value in zip(output_value, target_value):
            psnr_val_rgb.append(psnr(output_value, target_value))
            # output_value=output_value.unsqueeze(0)
            # target_value=target_value.unsqueeze(0)

    psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

    print(psnr_val_rgb)
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
from Net import SwinIR,IENet
from dataset import *

net=IENet()
net=net.cuda()
net.load_state_dict(torch.load('./model_n1_new.pth',weights_only=True))
inputPathTest="/home/liu/jxl/SOTS/outdoor/train_A/"
resultPathTest = '/home/liu/jxl/SWsr/n1_new/'  # 测试结果图片路径
datasetTest = TestDataSet(inputPathTest)
testLoader = DataLoader(dataset=datasetTest, batch_size=1)
net.eval()  # 指定网络模型测试状态
i=0
with torch.no_grad():
    timeStart = time.time()
    for index, (x,name) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
        input_test = x.cuda()
        output_test = net(input_test)
        
        print(name)
        #print(output_test, resultPathTest + str(index + 1).zfill(3) + ".png")
        save_image(output_test, resultPathTest + name[0])
        #save_image(input_test,resultPathTest + str(index + 1).zfill(3) + ".png")
    timeEnd = time.time()
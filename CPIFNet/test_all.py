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
from Net import *
from psnr import  psnr
from dataset import *
net1=IENet()
net2=IENet()
net3=IENet()
net1=net1.cuda()
net2=net2.cuda()

net1.load_state_dict(torch.load('./model_n1_new.pth'))
net2.load_state_dict(torch.load('./model_n1.pth'))


net4=IFNet()
net4=net4.cuda()

net4.load_state_dict(torch.load('./model_IFNet_new.pth'))

inputPathTest="/home/liu/jxl/realworld_hazy/"
resultPathTest = '/home/liu/jxl/SWsr/result/'  # 测试结果图片路径
datasetTest = TestDataSet(inputPathTest)
testLoader = DataLoader(dataset=datasetTest, batch_size=1)
net1.eval()  # 指定网络模型测试状态
net2.eval()

net4.eval()
with torch.no_grad():
    timeStart = time.time()
    for index, (x,name) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
        input_test = x.cuda()
        output_test1 = net1(input_test)
        output_test2=net2(input_test)

        
        output_test=net4(output_test1,output_test2)
        save_image(output_test1,resultPathTest +'/'+'n1_'+ name[0])
        save_image(output_test2,resultPathTest +'/'+'n2_' +name[0])
        #print(output_test, resultPathTest + str(index + 1).zfill(3) + ".png")
        save_image(output_test, resultPathTest +'/'+'fusion_' +name[0])
        #save_image(input_test,resultPathTest + str(index + 1).zfill(3) + ".png")
    timeEnd = time.time()
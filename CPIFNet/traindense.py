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
#from Net import IENet
from Net import *
from dataset import *
from psnr import  psnr
from Loss import *
Epoch=100
batch_size=22
best_ssim=0
learning_rate=1e-3
loss_list=[]
best_psnr=0
best_epoch=0

# inputPathTrain="F:\\dataset\\FiveK-Haze\\1117dense\\train_haze"
# targetPathTrain="F:\\dataset\\FiveK-Haze\\1117mist\\gt"
#
# inputPathTest="F:\\dataset\\FiveK-Haze\\1117dense\\test_haze"
# targetPathTest="F:\\dataset\\FiveK-Haze\\1117mist\\gt"

inputPathTrain="/home/liu/jxl/FiveK-Haze/1117dense/train_haze"
targetPathTrain="/home/liu/jxl/FiveK-Haze/1117mist/gt"
#
inputPathTest="/home/liu/jxl/FiveK-Haze/1117dense/test_haze"
targetPathTest="/home/liu/jxl/FiveK-Haze/1117mist/gt"

net=IENet()
net=net.cuda()
criterion1=nn.L1Loss()
criterion3=VGG()
criterion3=criterion3.cuda()
criterion2=MS_SSIM()
optimizer=optim.Adam(net.parameters(),lr=learning_rate,betas=(0.9,0.999),eps=10e-8)
datasetTrain=TrainDataSet(inputPathTrain,targetPathTrain)
scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90], gamma=0.5)
trainLoader=DataLoader(dataset=datasetTrain,batch_size=batch_size)

datasetValue=ValueDataSet(inputPathTest,targetPathTest)

valueLoader=DataLoader(dataset=datasetValue,batch_size=16)

datasetTest=TestDataSet(inputPathTest)
epochLoss_best=10000000
accumulation_steps = 4
testLoader=DataLoader(dataset=datasetTest,batch_size=1)
print("--------------------开始训练----------------------\n")
if os.path.exists('./model_mist.pth'):
    net.load_state_dict(torch.load('./model_mist.pth'))
for epoch in range(Epoch):
    net.train()
    iters=tqdm(trainLoader,file=sys.stdout)
    epochLoss=0
    psnr_sum=0
    accumulated_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
    timeStart=time.time()
    for index,(x,y) in enumerate(iters,0):
        net.zero_grad()
        optimizer.zero_grad()
        input_train,target=Variable(x).cuda(),Variable(y).cuda()
        output_train=net(input_train)
        #loss=criterion1(output_train,target)+color_loss(output_train,target)/100+criterion2(output_train,target)/5

        loss=criterion1(output_train,target)+ criterion3(output_train,target)
        loss.backward()
        optimizer.step()
        epochLoss+=loss.item()
        iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, Epoch, loss.item()))
    net.eval()
    psnr_val_rgb = []

    scheduler.step()
    epochLoss_test=0
    for index, (x, y) in enumerate(valueLoader, 0):
        input_, target_value = x.cuda(), y.cuda()
        with torch.no_grad():
            output_value = net(input_)
            loss = criterion1(output_value, target_value)   + criterion3(output_value, target_value)
            epochLoss_test += loss.item()
            psnr_sum=psnr_sum+psnr(output_value,target_value)
                #output_value=output_value.unsqueeze(0)
                #target_value=target_value.unsqueeze(0)


    psnr_sum=psnr_sum/len(valueLoader)

    print(psnr_sum)
    psnr_sum=0
    print(epochLoss_test/len(valueLoader))
    if epochLoss_test < epochLoss_best:
        epochLoss_best = epochLoss_test
        best_epoch = epoch
        torch.save(net.state_dict(), 'model_mist.pth')
        loss_list.append(epochLoss)  # 插入每次训练的损失值
        torch.save(net.state_dict(), 'model_mist.pth')  # 每次训练结束保存模型参数
        timeEnd = time.time()  # 每次训练结束时间
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}.".format(epoch+1, timeEnd-timeStart, epochLoss))
        print('-------------------------------------------------------------------------------------------------------')
print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))
print('-------------------------------------------------------------------------------------------------------')






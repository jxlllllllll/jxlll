import sys
import time
import numpy as np
from SSIM import *
import torch
import torch.nn as nn
import torch.optim as optim
#from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
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
batch_size=8
learning_rate=1e-3
loss_list=[]
best_psnr=0
best_epoch=0

inputPathTrain="/home/liu/jxl/SOTS/outdoor/train_A/"
targetPathTrain="/home/liu/jxl/densetrain_16_256/hr_256/"
#
# inputPathTest="/home/liu/jxl/densetrain_16_256/val/"
# targetPathTest="/home/liu/jxl/densetrain_16_256/hr_256/"

net1=IENet()
net2=IENet()
net3=IENet()
net1=net1.cuda()
net2=net2.cuda()
net3=net3.cuda()
net1.load_state_dict(torch.load('./model_n1_new.pth'))
net2.load_state_dict(torch.load('./model_n1.pth'))

net4=IFNet()
net4=net4.cuda()

criterion1=nn.L1Loss()
epochLoss_best=100000000

optimizer=optim.Adam(net4.parameters(),lr=learning_rate,betas=(0.9,0.999),eps=10e-8)

datasetTrain=TrainDataSet(inputPathTrain,targetPathTrain)

trainLoader=DataLoader(dataset=datasetTrain,batch_size=batch_size)

# datasetValue=ValueDataSet(dataset=datasetTrain,targetPathTest)

valueLoader=DataLoader(datasetTrain,batch_size=1)

# datasetTest=TestDataSet(inputPathTest)

# testLoader=DataLoader(dataset=datasetTest,batch_size=1)
print("--------------------开始训练----------------------\n")

net1.eval()
net2.eval()
net4.train()

if os.path.exists('./model_IFNet_best_new.pth'):
    net4.load_state_dict(torch.load('./model_IFNet_best_new.pth'))
for epoch in range(Epoch):
    if(epoch!=0 and epoch%10==0):
        learning_rate=learning_rate*0.9

    net4.train()
    iters=tqdm(trainLoader,file=sys.stdout)
    epochLoss=0
    timeStart=time.time()
    for index,(x,y) in enumerate(iters,0):
        net4.zero_grad()
        optimizer.zero_grad()
        input_train,target=Variable(x).cuda(),Variable(y).cuda()
        output_1=net1(input_train)
        output_2=net2(input_train)


        output_all=net4(output_1,output_2)

        loss=criterion1(output_all,target)
        loss.backward()
        optimizer.step()
        epochLoss+=loss.item()
        iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, Epoch, loss.item()))
    net4.eval()
    epochLoss_test=0
    psnr_sum=0
    for index, (x, y) in enumerate(valueLoader, 0):
        input_, target_value = x.cuda(), y.cuda()
        with torch.no_grad():
            output_value_1=net1(input_)
            output_value_2=net2(input_)
            output_all=net4(output_value_1,output_value_2)
            loss = criterion1(output_all, target_value)
            epochLoss_test += loss.item()
            psnr_sum=psnr_sum+psnr(output_all,target_value)
                #output_value=output_value.unsqueeze(0)
                #target_value=target_value.unsqueeze(0)


    psnr_sum=psnr_sum/len(valueLoader)

    print(psnr_sum)
    psnr_sum=0
    print(epochLoss_test/len(valueLoader))
    if epochLoss_test < epochLoss_best:
        epochLoss_best = epochLoss_test
        best_epoch = epoch
        torch.save(net4.state_dict(), 'model_IFNet_best_new.pth')
        loss_list.append(epochLoss)  # 插入每次训练的损失值
        torch.save(net4.state_dict(), 'model_IFNet_new.pth')  # 每次训练结束保存模型参数
        timeEnd = time.time()  # 每次训练结束时间
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}.".format(epoch+1, timeEnd-timeStart, epochLoss))
        print('-------------------------------------------------------------------------------------------------------')
print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))
print('-------------------------------------------------------------------------------------------------------')






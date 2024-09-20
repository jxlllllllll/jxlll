import math
import numpy as np
import torch
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torch.nn.functional as F
import torch
from torch import nn as nn
from torchvision import models
from torch.autograd import Variable
def rgb_to_v(rgb):
    # 归一化 RGB 值
    r, g, b = rgb[:, 0] / 255.0, rgb[:, 1] / 255.0, rgb[:, 2] / 255.0

    cmax = torch.max(r, torch.max(g, b))


    # 计算 V
    v = cmax

    return v

class L_m(nn.Module):
    def __init__(self):
        super(L_m, self).__init__()
    def forward(self,input,target):
        loss=torch.mean(torch.abs(input-target))
        return loss
class L_dcp(nn.Module):
    def __init__(self):
        super(L_dcp, self).__init__()
    def forward(self,input):
        r = input[:, 0, :, :]
        g = input[:, 1, :, :]
        b = input[:, 2, :, :]
        min_channel = torch.min(torch.min(r, g), b)
        dark_channel = torch.nn.functional.avg_pool2d(min_channel, kernel_size=15, stride=1,
                                                      padding=15 // 2)
        loss=torch.mean(dark_channel)
        return loss
class L_exp(nn.Module):
    def __init__(self):
        super(L_exp, self).__init__()
        self.avg=nn.AvgPool2d(2,stride=2)
    def forward(self,input):
        v=rgb_to_v(input)
        v=self.avg(v)
        out=torch.abs(v-0.5)
        loss=torch.mean(out)
        return loss
def color_loss(input_image, output_image):
    vec1 = input_image.view([-1, 3])
    vec2 = output_image.view([-1, 3])
    clip_value = 0.999999
    norm_vec1 = torch.nn.functional.normalize(vec1)
    norm_vec2 = torch.nn.functional.normalize(vec2)
    dot = norm_vec1 * norm_vec2
    dot = dot.mean(dim=1)
    dot = torch.clamp(dot, -clip_value, clip_value)
    angle = torch.acos(dot) * (180 / math.pi)
    return angle.mean()


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h_relu_1_2 = self.to_relu_1_2(x)
        h_relu_2_2 = self.to_relu_2_2(h_relu_1_2)
        h_relu_3_3 = self.to_relu_3_3(h_relu_2_2)
        h_relu_4_3 = self.to_relu_4_3(h_relu_3_3)
        return [h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3]
class MAE(nn.Module):
    def __init__(self, size_average=True):
        super(MAE, self).__init__()
        self.size_average = size_average

    def forward(self, x1, x2):
        if self.size_average:
            return abs(x1 - x2).mean()
        else:
            return abs(x1 - x2)
class VGG_loss(nn.Module):
    def __init__(self, size_average=True):
        super(VGG_loss, self).__init__()
        # self.vgg = Vgg16().type(torch.cuda.FloatTensor)
        self.VGG = VGG16()
        self.MAE = MAE(size_average)
        self.size_average = size_average

    def forward(self, hazy, gth):
        hazy = self.VGG(hazy)
        gth = self.VGG(gth)
        out_1 = self.MAE(hazy[0], gth[0])
        out_2 = self.MAE(hazy[1], gth[1])
        out_3 = self.MAE(hazy[2], gth[2])
        out_4 = self.MAE(hazy[3], gth[3])
        if self.size_average:
            return (out_1 + out_2 + out_3 + out_4) / 4
        else:
            return [out_1, out_2, out_3, out_4]
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
class MS_SSIM(torch.nn.Module):
    def __init__(self, size_average=True, max_val=255):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val

    def _ssim(self, img1, img2, size_average=True):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=self.channel) - mu1_mu2

        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())

        msssim = Variable(torch.Tensor(levels, ).cuda())
        mcs = Variable(torch.Tensor(levels, ).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels - 1] ** weight[0:levels - 1]) *
                 (msssim[levels - 1] ** weight[levels - 1]))
        return value

    def forward(self, img1, img2):

        return self.ms_ssim(img1, img2)
class color_loss(nn.Module):
    def __init__(self):
        super(color_loss, self).__init__()

    def forward(self, x, y):
        b, c, h, w = x.shape

        mr_x, mg_x, mb_x = torch.split(x, 1, dim=1)
        mr_x, mg_x, mb_x = mr_x.view([b, 1, -1, 1]), mg_x.view([b, 1, -1, 1]), mb_x.view([b, 1, -1, 1])
        xx = torch.cat([mr_x, mg_x, mb_x], dim=3).squeeze(1) + 0.000001

        mr_y, mg_y, mb_y = torch.split(y, 1, dim=1)
        mr_y, mg_y, mb_y = mr_y.view([b, 1, -1, 1]), mg_y.view([b, 1, -1, 1]), mb_y.view([b, 1, -1, 1])
        yy = torch.cat([mr_y, mg_y, mb_y], dim=3).squeeze(1) + 0.000001
        
        xx = xx.reshape(h * w * b, 3)
        yy = yy.reshape(h * w * b, 3)
        l_x = torch.sqrt(pow(xx[:, 0], 2) + pow(xx[:, 1], 2) + pow(xx[:, 2], 2))
        l_y = torch.sqrt(pow(yy[:, 0], 2) + pow(yy[:, 1], 2) + pow(yy[:, 2], 2))
        xy = xx[:, 0] * yy[:, 0] + xx[:, 1] * yy[:, 1] + xx[:, 2] * yy[:, 2]
        cos_angle = xy / (l_x * l_y + 0.000001)
        angle = torch.acos(cos_angle.cpu())
        angle2 = angle * 360 / 2 / np.pi
        # re = angle2.reshape(b, -1)
        an_mean = torch.mean(angle2) / 180
        return an_mean.cuda()

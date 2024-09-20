import math

import torch
from torch import nn


def psnr(x,y):
    mse_loss = nn.MSELoss()(x * 0.5 + 0.5, y * 0.5 + 0.5)
    psnr = 10 * math.log10(1 / mse_loss)
    return psnr

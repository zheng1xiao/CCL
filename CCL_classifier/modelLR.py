import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.autograd import Variable
import numpy as np

#
#
#
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.lr=nn.Linear(14,1)   #相当于通过线性变换y=x*T(A)+b可以得到对应的各个系数
        self.sm=nn.Sigmoid()   #相当于通过激活函数的变换

    def forward(self, x):
        x=self.lr(x)
        x=self.sm(x)
        return x





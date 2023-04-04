import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.autograd import Variable
import numpy as np

#
#
#
# class LogisticRegression1(nn.Module):
#     def __init__(self):
#         super(LogisticRegression1,self).__init__()
#         self.lr=nn.Linear(14,1)   #相当于通过线性变换y=x*T(A)+b可以得到对应的各个系数
#         self.sm=nn.Sigmoid()   #相当于通过激活函数的变换
#
#     def forward(self, x):
#         x=self.lr(x)
#         x=self.sm(x)
#         return x


# num_i = 28 * 28  # 输入层节点数
# num_h = 100  # 隐含层节点数
# num_o = 10  # 输出层节点数
# batch_size = 64


class MLP(nn.Module):

    def __init__(self, num_i, num_h1, num_h2, num_o):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(num_i, num_h1)
        # self.relu = nn.ReLU()
        self.lkrelu1 = nn.LeakyReLU(0.05)
        # self.bn1 = nn.BatchNorm1d(num_h)
        self.linear2 = nn.Linear(num_h1, num_h2)  # 2个隐层
        # self.relu2 = nn.ReLU()
        self.lkrelu2 = nn.LeakyReLU(0.05)
        # self.bn2 = nn.BatchNorm1d(num_h)
        self.linear3 = nn.Linear(num_h2, num_o)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.linear1(x)
        x = self.lkrelu1(x)
        # x = self.relu(x)
        # x = self.bn1(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        x = self.lkrelu2(x)
        # x = self.relu2(x)
        # x = self.bn2(x)
        # x = self.dropout(x)
        x = self.linear3(x)
        return x



import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.autograd import Variable
import numpy as np




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



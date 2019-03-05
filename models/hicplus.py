""" Model for enhance HiC matrix """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    """ the convolutional net from paper """
    def __init__(self, D_in, D_out):
        super(ConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernels
        self.conv1 = nn.Conv2d(1, 8, 9, stride=1)
        self.conv2 = nn.Conv2d(8, 8, 1)
        self.conv3 = nn.Conv2d(8, 1, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x


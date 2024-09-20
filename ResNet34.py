# -*- coding: utf-8 -*-
import torch
from torchvision.models.resnet import resnet34
import torch.nn as nn
from collections import OrderedDict
import torch
import math


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        # kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        # kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size = 3

        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)

        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return x * v


class MyResNetD(nn.Module):
    # OutChannal represents kernal size.
    def __init__(self, num_class=1000):
        super(MyResNetD, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer1 0
        self.layer10conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer10bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer10relu = nn.ReLU(inplace=True)
        self.layer10conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer10bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1 1
        self.layer11conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer11bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer11relu = nn.ReLU(inplace=True)
        self.layer11conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer11bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1 2
        self.layer12conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer12bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer12relu = nn.ReLU(inplace=True)
        self.layer12conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer12bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2 0
        self.layer20conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer20bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer20relu = nn.ReLU(inplace=True)
        self.layer20conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer20bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer20downsample0 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer20downsample1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2 1
        self.layer21conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer21bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer21relu = nn.ReLU(inplace=True)
        self.layer21conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer21bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2 2
        self.layer22conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer22bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer22relu = nn.ReLU(inplace=True)
        self.layer22conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer22bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # layer2 3
        self.layer23conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer23bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer23relu = nn.ReLU(inplace=True)
        self.layer23conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer23bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3 0
        self.layer30conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer30bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer30relu = nn.ReLU(inplace=True)
        self.layer30conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer30bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer30downsample0 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer30downsample1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer 3 1
        self.layer31conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer31bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer31relu = nn.ReLU(inplace=True)
        self.layer31conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer31bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer 3 2
        self.layer32conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer32relu = nn.ReLU(inplace=True)
        self.layer32conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # layer 3 3
        self.layer33conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer33bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer33relu = nn.ReLU(inplace=True)
        self.layer33conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer33bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer 3 4
        self.layer34conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer34bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer34relu = nn.ReLU(inplace=True)
        self.layer34conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer34bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer 3 5
        self.layer35conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer35bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer35relu = nn.ReLU(inplace=True)
        self.layer35conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer35bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer 4 0
        self.layer40conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer40bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer40relu = nn.ReLU(inplace=True)
        self.layer40conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer40bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer40downsample0 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer40downsample1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer 4 1
        self.layer41conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer41bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer41relu = nn.ReLU(inplace=True)
        self.layer41conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer41bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # layer 4 2
        self.layer42conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer42bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer42relu = nn.ReLU(inplace=True)
        self.layer42conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer42bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layer1 0
        identity = x
        x = self.layer10conv1(x)
        x = self.layer10bn1(x)
        x = self.layer10relu(x)
        x = self.layer10conv2(x)
        x = self.layer10bn2(x)
        x += identity
        x = self.layer10relu(x)

        # layer1 1
        identity = x
        x = self.layer11conv1(x)
        x = self.layer11bn1(x)
        x = self.layer11relu(x)
        x = self.layer11conv2(x)
        x = self.layer11bn2(x)
        x += identity
        x = self.layer11relu(x)

        # layer1 2
        identity = x
        x = self.layer12conv1(x)
        x = self.layer12bn1(x)
        x = self.layer12relu(x)  # relu
        x = self.layer12conv2(x)
        x = self.layer12bn2(x)
        x += identity
        x = self.layer12relu(x)  # relu

        # layer2 0
        identity = x
        x = self.layer20conv1(x)
        x = self.layer20bn1(x)
        x = self.layer20relu(x)  # relu
        x = self.layer20conv2(x)
        x = self.layer20bn2(x)
        identity = self.layer20downsample0(identity)
        identity = self.layer20downsample1(identity)
        x += identity
        x = self.layer20relu(x)  # relu

        # layer2 1
        identity = x
        x = self.layer21conv1(x)
        x = self.layer21bn1(x)
        x = self.layer21relu(x)  # relu
        x = self.layer21conv2(x)
        x = self.layer21bn2(x)
        x += identity
        x = self.layer20relu(x)  # relu

        # layer2 2
        identity = x
        x = self.layer22conv1(x)
        x = self.layer22bn1(x)
        x = self.layer22relu(x)  # relu
        x = self.layer22conv2(x)
        x = self.layer22bn2(x)
        x += identity
        x = self.layer22relu(x)  # relu
        # layer2 3
        identity = x
        x = self.layer23conv1(x)
        x = self.layer23bn1(x)
        x = self.layer23relu(x)  # relu
        x = self.layer23conv2(x)
        x = self.layer23bn2(x)
        x += identity
        x = self.layer23relu(x)  # relu

        # layer 3 0
        identity = x
        x = self.layer30conv1(x)
        x = self.layer30bn1(x)
        x = self.layer30relu(x)  # relu
        x = self.layer30conv2(x)
        x = self.layer30bn2(x)
        identity = self.layer30downsample0(identity)
        identity = self.layer30downsample1(identity)
        x += identity
        x = self.layer30relu(x)  # relu

        # layer 3 1
        identity = x
        x = self.layer31conv1(x)
        x = self.layer31bn1(x)
        x = self.layer31relu(x)  # relu
        x = self.layer31conv2(x)
        x = self.layer31bn2(x)
        x += identity
        x = self.layer31relu(x)  # relu

        # layer 3 2
        identity = x
        x = self.layer32conv1(x)
        x = self.layer32bn1(x)
        x = self.layer32relu(x)  # relu
        x = self.layer32conv2(x)
        x = self.layer32bn2(x)
        x += identity
        x = self.layer32relu(x)  # relu

        # layer 3 3
        identity = x
        x = self.layer33conv1(x)
        x = self.layer33bn1(x)
        x = self.layer33relu(x)  # relu
        x = self.layer33conv2(x)
        x = self.layer33bn2(x)
        x += identity
        x = self.layer33relu(x)  # relu

        # layer 3 4
        identity = x
        x = self.layer34conv1(x)
        x = self.layer34bn1(x)
        x = self.layer34relu(x)  # relu
        x = self.layer34conv2(x)
        x = self.layer34bn2(x)
        x += identity
        x = self.layer34relu(x)  # relu
        # layer 3 5
        identity = x
        x = self.layer35conv1(x)
        x = self.layer35bn1(x)
        x = self.layer35relu(x)  # relu
        x = self.layer35conv2(x)
        x = self.layer35bn2(x)
        x += identity
        x = self.layer35relu(x)  # relu

        # layer 4 0
        identity = x
        x = self.layer40conv1(x)
        x = self.layer40bn1(x)
        x = self.layer40relu(x)  # relu
        x = self.layer40conv2(x)
        x = self.layer40bn2(x)
        identity = self.layer40downsample0(identity)
        identity = self.layer40downsample1(identity)
        x += identity
        x = self.layer40relu(x)  # relu

        # layer 4 1
        identity = x
        x = self.layer41conv1(x)
        x = self.layer41bn1(x)
        x = self.layer41relu(x)  # relu
        x = self.layer41conv2(x)
        x = self.layer41bn2(x)
        x += identity
        x = self.layer41relu(x)  # relu
        # layer 4 2
        identity = x
        x = self.layer42conv1(x)
        x = self.layer42bn1(x)
        x = self.layer42relu(x)  # relu
        x = self.layer42conv2(x)
        x = self.layer42bn2(x)
        x += identity
        x = self.layer42relu(x)  # relu

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class MyResNetD_Plus(nn.Module):
    # OutChannal represents kernal size.
    def __init__(self, num_class=1000):
        super(MyResNetD_Plus, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer1 0
        self.layer10conv1 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer10bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer10relu = nn.ReLU(inplace=True)
        self.layer10conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer10bn2 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1 1
        self.layer11conv1 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer11bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer11relu = nn.ReLU(inplace=True)
        self.layer11conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer11bn2 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer1att = ECABlock(16)

        # layer1 2
        self.layer12conv1 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer12bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer12relu = nn.ReLU(inplace=True)
        self.layer12conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer12bn2 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2 0
        self.layer20conv1 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer20bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer20relu = nn.ReLU(inplace=True)
        self.layer20conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer20bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer20downsample0 = nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer20downsample1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2 1
        self.layer21conv1 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer21bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer21relu = nn.ReLU(inplace=True)
        self.layer21conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer21bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2 2
        self.layer22conv1 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer22bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer22relu = nn.ReLU(inplace=True)
        self.layer22conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer22bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # layer2 3
        self.layer23conv1 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer23bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer23relu = nn.ReLU(inplace=True)
        self.layer23conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer23bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2att = ECABlock(32)

        # layer3 0
        self.layer30conv1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer30bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer30relu = nn.ReLU(inplace=True)
        self.layer30conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer30bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer30downsample0 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer30downsample1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer 3 1
        self.layer31conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer31bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer31relu = nn.ReLU(inplace=True)
        self.layer31conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer31bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer 3 2
        self.layer32conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer32relu = nn.ReLU(inplace=True)
        self.layer32conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # layer 3 3
        self.layer33conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer33bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer33relu = nn.ReLU(inplace=True)
        self.layer33conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer33bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer 3 4
        self.layer34conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer34bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer34relu = nn.ReLU(inplace=True)
        self.layer34conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer34bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer 3 5
        self.layer35conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer35bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer35relu = nn.ReLU(inplace=True)
        self.layer35conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer35bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3att = ECABlock(64)

        # layer 4 0
        self.layer40conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer40bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer40relu = nn.ReLU(inplace=True)
        self.layer40conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer40bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer40downsample0 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer40downsample1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer 4 1
        self.layer41conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer41bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer41relu = nn.ReLU(inplace=True)
        self.layer41conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer41bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # layer 4 2
        self.layer42conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer42bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer42relu = nn.ReLU(inplace=True)
        self.layer42conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer42bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4att = ECABlock(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layer1 0
        identity = x
        x = self.layer10conv1(x)
        x = self.layer10bn1(x)
        x = self.layer10relu(x)
        x = self.layer10conv2(x)
        x = self.layer10bn2(x)
        x += identity
        x = self.layer10relu(x)

        # layer1 1
        identity = x
        x = self.layer11conv1(x)
        x = self.layer11bn1(x)
        x = self.layer11relu(x)
        x = self.layer11conv2(x)
        x = self.layer11bn2(x)
        x += identity
        x = self.layer11relu(x)

        x = self.layer1att(x)

        # layer1 2
        identity = x
        x = self.layer12conv1(x)
        x = self.layer12bn1(x)
        x = self.layer12relu(x)  # relu
        x = self.layer12conv2(x)
        x = self.layer12bn2(x)
        x += identity
        x = self.layer12relu(x)  # relu

        # layer2 0
        identity = x
        x = self.layer20conv1(x)
        x = self.layer20bn1(x)
        x = self.layer20relu(x)  # relu
        x = self.layer20conv2(x)
        x = self.layer20bn2(x)
        identity = self.layer20downsample0(identity)
        identity = self.layer20downsample1(identity)
        x += identity
        x = self.layer20relu(x)  # relu

        # layer2 1
        identity = x
        x = self.layer21conv1(x)
        x = self.layer21bn1(x)
        x = self.layer21relu(x)  # relu
        x = self.layer21conv2(x)
        x = self.layer21bn2(x)
        x += identity
        x = self.layer20relu(x)  # relu

        # layer2 2
        identity = x
        x = self.layer22conv1(x)
        x = self.layer22bn1(x)
        x = self.layer22relu(x)  # relu
        x = self.layer22conv2(x)
        x = self.layer22bn2(x)
        x += identity
        x = self.layer22relu(x)  # relu
        # layer2 3
        identity = x
        x = self.layer23conv1(x)
        x = self.layer23bn1(x)
        x = self.layer23relu(x)  # relu
        x = self.layer23conv2(x)
        x = self.layer23bn2(x)
        x += identity
        x = self.layer23relu(x)  # relu
        x = self.layer2att(x)

        # layer 3 0
        identity = x
        x = self.layer30conv1(x)
        x = self.layer30bn1(x)
        x = self.layer30relu(x)  # relu
        x = self.layer30conv2(x)
        x = self.layer30bn2(x)
        identity = self.layer30downsample0(identity)
        identity = self.layer30downsample1(identity)
        x += identity
        x = self.layer30relu(x)  # relu

        # layer 3 1
        identity = x
        x = self.layer31conv1(x)
        x = self.layer31bn1(x)
        x = self.layer31relu(x)  # relu
        x = self.layer31conv2(x)
        x = self.layer31bn2(x)
        x += identity
        x = self.layer31relu(x)  # relu

        # layer 3 2
        identity = x
        x = self.layer32conv1(x)
        x = self.layer32bn1(x)
        x = self.layer32relu(x)  # relu
        x = self.layer32conv2(x)
        x = self.layer32bn2(x)
        x += identity
        x = self.layer32relu(x)  # relu

        # layer 3 3
        identity = x
        x = self.layer33conv1(x)
        x = self.layer33bn1(x)
        x = self.layer33relu(x)  # relu
        x = self.layer33conv2(x)
        x = self.layer33bn2(x)
        x += identity
        x = self.layer33relu(x)  # relu

        # layer 3 4
        identity = x
        x = self.layer34conv1(x)
        x = self.layer34bn1(x)
        x = self.layer34relu(x)  # relu
        x = self.layer34conv2(x)
        x = self.layer34bn2(x)
        x += identity
        x = self.layer34relu(x)  # relu
        # layer 3 5
        identity = x
        x = self.layer35conv1(x)
        x = self.layer35bn1(x)
        x = self.layer35relu(x)  # relu
        x = self.layer35conv2(x)
        x = self.layer35bn2(x)
        x += identity
        x = self.layer35relu(x)  # relu
        x = self.layer3att(x)

        # layer 4 0
        identity = x
        x = self.layer40conv1(x)
        x = self.layer40bn1(x)
        x = self.layer40relu(x)  # relu
        x = self.layer40conv2(x)
        x = self.layer40bn2(x)
        identity = self.layer40downsample0(identity)
        identity = self.layer40downsample1(identity)
        x += identity
        x = self.layer40relu(x)  # relu

        # layer 4 1
        identity = x
        x = self.layer41conv1(x)
        x = self.layer41bn1(x)
        x = self.layer41relu(x)  # relu
        x = self.layer41conv2(x)
        x = self.layer41bn2(x)
        x += identity
        x = self.layer41relu(x)  # relu
        # layer 4 2
        identity = x
        x = self.layer42conv1(x)
        x = self.layer42bn1(x)
        x = self.layer42relu(x)  # relu
        x = self.layer42conv2(x)
        x = self.layer42bn2(x)
        x += identity
        x = self.layer42relu(x)  # relu
        x = self.layer4att(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


from torchvision import models

# Onet = models.resnet34(pretrained=True)  # 官方模型，包含了预训练参数
# OWeight = Onet.state_dict()

net = MyResNetD()  # 自己的模型，参数名称变化了
# Cweight = net.state_dict()

# for k, v in OWeight.items():
#     if k in Cweight.keys():  # 加载相同的参数名称
#         Cweight[k] = v
#     if k.startswith('layer') and 'downsample' not in k:  # 加载代码生成的layer但
#         kt = k.replace('.', '', 2)  # 不包含downsample的名称
#         Cweight[kt] = v
#     if 'downsample' in k:  # 加载downsample层名称的参数
#         kt = k.replace('.', '', 3)
#         Cweight[kt] = v
# net.load_state_dict(Cweight)

# out = torch.rand(10, 3, 224, 224)
# # 对比输出结果
# out2 = net(out)
# outO = Onet(out)
# print(out2[1, 0:10])
# print(outO[1, 0:10])

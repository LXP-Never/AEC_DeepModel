# Author:凌逆战
# -*- coding:utf-8 -*-
"""
作用：
"""
import torch.nn as nn
import torch

# Author:凌逆战
# -*- coding:utf-8 -*-
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils import weight_norm
from torch.nn import functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, short_channel_len, short_width = x.size()

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCN_block(nn.Module):
    """这个padding和我的有些不同"""

    def __init__(self, in_channel, out_channel, kernel_size, dilation):
        super(TCN_block, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(num_features=out_channel)  # BN有bias的作用
        self.LeakyReLU1 = nn.LeakyReLU(negative_slope=0.2)
        # self.dropout1 = nn.Dropout(dropout)
        # ---------------------------------------------------------------
        self.conv2 = nn.Conv1d(out_channel, out_channel * 2, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_features=out_channel * 2)
        # ---------------------------------------------------------------
        if in_channel == 2 * out_channel:
            self.downsample = None
        else:
            self.downsample = nn.Conv1d(in_channel, out_channel * 2, kernel_size=1)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.init_weights()

    def init_weights(self):
        init.orthogonal_(self.conv1.weight)
        init.zeros_(self.conv1.bias)
        init.orthogonal_(self.conv2.weight)
        init.zeros_(self.conv2.bias)
        # BN层
        init.normal_(self.bn1.weight, mean=1.0, std=0.02)
        init.constant_(self.bn1.bias, 0)
        init.normal_(self.bn2.weight, mean=1.0, std=0.02)
        init.constant_(self.bn2.bias, 0)
        if self.downsample is not None:
            init.orthogonal_(self.downsample.weight)

    def forward(self, input):
        x = self.conv1(input)
        x = self.chomp1(x)
        x = self.bn1(x)
        x = self.LeakyReLU1(x)
        # --------------------------
        x = self.conv2(x)
        out = self.bn2(x)

        if self.downsample is None:
            res = input
        else:
            res = self.downsample(input)
        return self.LeakyReLU(out + res)


class TCN_block_k3(nn.Module):
    # 如果用这个kernel必须设置为3
    def __init__(self, in_channel, out_channel, kernel_size, dilation):
        super(TCN_block_k3, self).__init__()
        self.padding = nn.ReflectionPad1d(dilation)
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(num_features=out_channel)  # BN有bias的作用
        self.leakyrelu_1 = nn.LeakyReLU(negative_slope=0.2)
        # self.dropout1 = nn.Dropout(dropout)
        # ---------------------------------------------------------------
        self.conv2 = nn.Conv1d(out_channel, out_channel * 2, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=out_channel * 2)
        # ---------------------------------------------------------------
        if in_channel == 2 * out_channel:
            self.downsample = None
        else:
            self.downsample = nn.Conv1d(in_channel, out_channel * 2, kernel_size=1)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.init_weights()

    def init_weights(self):
        init.orthogonal_(self.conv1.weight)  # 第一层卷积权重初始化
        init.orthogonal_(self.conv2.weight)  # 第二层卷积权重初始化
        # BN层
        init.normal_(self.bn1.weight, mean=1.0, std=0.02)
        init.constant_(self.bn1.bias, 0)
        init.normal_(self.bn2.weight, mean=1.0, std=0.02)
        init.constant_(self.bn2.bias, 0)
        if self.downsample is not None:
            init.orthogonal_(self.downsample.weight)

    def forward(self, input):
        x = self.padding(input)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu_1(x)
        # --------------------------
        x = self.conv2(x)
        out = self.bn2(x)

        if self.downsample is None:
            # print("1")
            res = input
        else:
            # print("2")
            res = self.downsample(input)
        return self.LeakyReLU(out + res)


class TCN_model(nn.Module):
    def __init__(self):
        super(TCN_model, self).__init__()
        # (64, 322, 998)
        self.first = nn.Conv1d(in_channels=322, out_channels=161, kernel_size=9, stride=2, padding=9 // 2)  # (64, 161, 499)
        self.TCN_conv0 = TCN_block(in_channel=161, out_channel=161, kernel_size=9, dilation=2 ** 0)
        self.TCN_conv1 = TCN_block(in_channel=322, out_channel=161, kernel_size=9, dilation=2 ** 1)
        self.TCN_conv2 = TCN_block(in_channel=322, out_channel=161, kernel_size=9, dilation=2 ** 2)

        self.lastconv = nn.Conv1d(in_channels=322, out_channels=2, kernel_size=9, stride=1, padding=9 // 2)
        self.subpix = PixelShuffle1D(upscale_factor=2)
        self.init_weights()

    def init_weights(self):
        init.orthogonal_(self.first.weight)
        init.constant_(self.first.bias, 0)
        init.normal_(self.lastconv.weight, mean=0, std=1e-3)
        init.constant_(self.lastconv.bias, 0)

    def forward(self, input):
        # inputs    (64, 322, 999)
        x = self.first(input)       # torch.Size([64, 161, 500])
        print(x.shape)
        x = self.TCN_conv0(x)       # torch.Size([64, 322, 500])
        print(x.shape)
        x = self.TCN_conv1(x)
        print(x.shape)
        x = self.TCN_conv2(x)
        print(x.shape)
        x = self.lastconv(x)
        print(x.shape)
        x = self.subpix(x)
        print(x.shape)
        return x


x = torch.randn(64, 322, 998)
model = TCN_model()
output = model(x)
print(output.shape)












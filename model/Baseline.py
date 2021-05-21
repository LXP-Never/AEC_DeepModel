# Author:凌逆战
# -*- coding:utf-8 -*-
"""
作用：随便搭建的模型，只要符合输入大小是[64, 322, 999]，输出大小是[64, 161, 999]，就能跑通
"""
import torch.nn as nn
import torch


class Base_model(nn.Module):
    def __init__(self):
        super(Base_model, self).__init__()
        # [batch, channel, input_size] (B, F, T)
        # [64, 322, 999] ---> [64, 161, 999]
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=322, out_channels=322, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=322, out_channels=322, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=322, out_channels=161, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=161, out_channels=161, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: 麦克风信号和远端信号的特征串联在一起作为输入特征 (322, 206)
        :return: IRM_mask * input = 近端语音对数谱
        """
        Estimated_IRM = self.model(x)

        return Estimated_IRM


if __name__ == "__main__":
    model = Base_model().cuda()
    x = torch.randn(8, 322, 999).to("cuda")  # 输入 [8, 322, 999]
    y = model(x)  # 输出 [8, 161, 999]
    print(y.shape)

# Author:凌逆战
# -*- coding:utf-8 -*-
"""
作用：
"""
import torch



def frequency_MSE_loss(logits, labels):
    """ 均方误差，频域损失
    labels:batch_labels
    logits:batch_logits
    """
    loss = torch.mean((logits - labels) ** 2)
    return loss


def frequency_RMSE_loss(logits, labels):
    """ 均方根误差，频域损失
    labels:batch_labels
    logits:batch_logits
    """
    loss = torch.sqrt(torch.mean((logits - labels) ** 2, dim=[1, 2]))
    loss = torch.mean(loss, dim=0)
    return loss


def frequency_MAE_loss(logits, labels):
    """ 平均绝对值误差，频域损失
    labels:batch_labels
    logits:batch_logits
    """
    loss = torch.mean(torch.abs(logits - labels))
    return loss

# ###################### 计算LSD  ######################
def pytorch_LSD(logits, labels):
    # (…, freq, time)

    logits_log = torch.log10(logits ** 2 + 3e-9)
    labels_log = torch.log10(labels ** 2 + 3e-9)
    original_target_squared = (labels_log - logits_log) ** 2

    # lsd = torch.mean(torch.sqrt(torch.mean(original_target_squared, dim=0)))
    lsd = torch.mean(torch.sqrt(torch.mean(original_target_squared, dim=1)), dim=1)
    lsd = torch.mean(lsd, dim=0)

    return lsd
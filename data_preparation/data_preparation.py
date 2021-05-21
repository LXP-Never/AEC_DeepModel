# Author:凌逆战
# -*- coding:utf-8 -*-
"""
作用：
"""
import glob
import os
import torch.nn.functional as F
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class FileDateset(Dataset):
    def __init__(self, dataset_path="./Synthetic/TRAIN", fs=16000, win_length=320, mode="train"):
        self.fs = fs
        self.win_length = win_length
        self.mode = mode

        farend_speech_path = os.path.join(dataset_path, "farend_speech")        # "./Synthetic/TRAIN/farend_speech"
        nearend_mic_signal_path = os.path.join(dataset_path, "nearend_mic_signal")  # "./Synthetic/TRAIN/nearend_mic_signal"
        nearend_speech_path = os.path.join(dataset_path, "nearend_speech")      # "./Synthetic/TRAIN/nearend_speech"

        self.farend_speech_list = sorted(glob.glob(farend_speech_path+"/*.wav"))    # 远端语音路径，list
        self.nearend_mic_signal_list = sorted(glob.glob(nearend_mic_signal_path+"/*.wav"))  # 近端麦克风语音路径，list
        self.nearend_speech_list = sorted(glob.glob(nearend_speech_path+"/*.wav"))  # 近端语音路径，list

    def spectrogram(self, wav_path):
        """
        :param wav_path: 音频路径
        :return: 返回该音频的振幅和相位
        """
        wav, _ = torchaudio.load(wav_path)
        wav = wav.squeeze()

        if len(wav) < 160000:
            wav = F.pad(wav, (0,160000-len(wav)), mode="constant",value=0)

        S = torch.stft(wav, n_fft=self.win_length, hop_length=self.win_length//2,
                       win_length=self.win_length, window=torch.hann_window(window_length=self.win_length),
                       center=False, return_complex=True)   # (*, F,T)
        magnitude = torch.abs(S)        # 振幅
        phase = torch.exp(1j * torch.angle(S))  # 相位
        return magnitude, phase


    def __getitem__(self, item):
        """__getitem__是类的专有方法，使类可以像list一样按照索引来获取元素
        :param item: 索引
        :return:  按 索引取出来的 元素
        """
        # 远端语音 振幅，相位 （F, T）,F为频点数，T为帧数
        farend_speech_magnitude, farend_speech_phase = self.spectrogram(self.farend_speech_list[item])  # torch.Size([161, 999])
        # 近端麦克风 振幅，相位
        nearend_mic_magnitude, nearend_mic_phase = self.spectrogram(self.nearend_mic_signal_list[item])
        # 近端语音 振幅，相位
        nearend_speech_magnitude, nearend_speech_phase = self.spectrogram(self.nearend_speech_list[item])

        X = torch.cat((farend_speech_magnitude, nearend_mic_magnitude), dim=0)  # 在频点维度上进行拼接(161*2, 999),模型输入

        _eps = torch.finfo(torch.float).eps  # 防止分母出现0
        mask_IRM = torch.sqrt(nearend_speech_magnitude ** 2/(nearend_mic_magnitude ** 2+_eps))  # IRM，模型输出


        return X, mask_IRM, nearend_mic_magnitude, nearend_speech_magnitude

    def __len__(self):
        """__len__是类的专有方法，获取整个数据的长度"""
        return len(self.farend_speech_list)


if __name__ == "__main__":
    train_set = FileDateset()
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)

    for x, y, nearend_mic_magnitude,nearend_speech_magnitude in train_loader:
        print(x.shape)  # torch.Size([64, 322, 999])
        print(y.shape)  # torch.Size([64, 161, 999])
        print(nearend_mic_magnitude.shape)

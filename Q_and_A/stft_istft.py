# Author:凌逆战
# -*- coding:utf-8 -*-
"""
作用：
"""
import librosa
import torch
import torchaudio
import matplotlib.pyplot as plt

y, fs = torchaudio.load("./p225_001.wav")
print(y.shape)  # torch.Size([1, 98473])
S = torch.stft(y, n_fft=128, hop_length=64, win_length=128,return_complex=True)
print(S.shape)  # [1, 65, 1539]
magnitude = torch.abs(S)
phase = torch.angle(S)


stft=magnitude*torch.exp(1j*phase)

wav = torch.istft(stft,n_fft=128,hop_length=64,win_length=128)
print(y.shape)  # torch.Size([1, 98432])


y =y.numpy().flatten()
recover_wav = wav.numpy().flatten()

plt.plot(y, label="y")
plt.plot(recover_wav,label="y_recov")
plt.legend()
plt.show()
# Author:凌逆战
# -*- coding:utf-8 -*-
"""
作用：
"""

import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt

y, fs = librosa.load("./p225_001.wav", sr=16000)


def norm_spectrogram(stft, w_size):
    MD = 80
    ep = 10 ** (-1 * MD / 20)

    stft_norm = 20 * np.log10(np.abs(stft) / (w_size / 2) + ep)
    stft_norm = stft_norm + MD
    stft_norm = stft_norm / MD
    stft_norm = np.round(stft_norm * 255) / 255

    return stft_norm


def denorm_spectrogram(stft_norm, w_size):
    MD = 80
    ep = 10 ** (-1 * MD / 20)

    stft = stft_norm * MD
    stft = stft - MD
    stft = (10 ** (stft / 20) - ep) * (w_size / 2)

    return stft


def spectrogram(s, fs):
    s = s.astype('float32')
    w_size_ms = 19.90
    w_size_n = int(w_size_ms * (1e-3) * fs)

    stft = librosa.stft(s, n_fft=w_size_n, hop_length=w_size_n//2, win_length=w_size_n)     # (160, 207)

    stft_m = norm_spectrogram(stft, w_size_n)   # (160, 207)
    stft_p = np.angle(stft)         # (160, 207)

    return stft_m, stft_p


def recover_from_spectrogram(stft_m, stft_p, fs):
    w_size_ms = 19.90
    w_size_n = int(w_size_ms * (1e-3) * fs)

    # Inverse operations to denormalize stft
    stft_m = denorm_spectrogram(stft_m, w_size_n)       # (160, 207)

    # Combining phase and signal
    stft = stft_m.astype(np.complex) * np.exp(1j * stft_p)
    # IFFT
    s = librosa.istft(stft, hop_length=w_size_n//2, win_length=w_size_n)

    return s


stft_m, stft_p = spectrogram(y, fs)

recover_wav = recover_from_spectrogram(stft_m, stft_p, fs)


print(y.shape)          # (32825,)
print(recover_wav.shape)    # (32785,)

plt.plot(y, label="y")
plt.plot(recover_wav,label="y_recov")
plt.legend()
plt.show()
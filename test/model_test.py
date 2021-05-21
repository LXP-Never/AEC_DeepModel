# Author:凌逆战
# -*- coding:utf-8 -*-
"""
作用：通过模型生成近端语音
"""
import librosa
import matplotlib
import torchaudio
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from model.Baseline import Base_model
from matplotlib.ticker import FuncFormatter
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号


def spectrogram(wav_path, win_length=320):
    wav, _ = torchaudio.load(wav_path)
    wav = wav.squeeze()

    if len(wav) < 160000:
        wav = F.pad(wav, (0, 160000 - len(wav)), mode="constant", value=0)
    # if len(wav) != 160000:
    #     print(wav_path)
    #     print(len(wav))

    S = torch.stft(wav, n_fft=win_length, hop_length=win_length // 2,
                   win_length=win_length, window=torch.hann_window(window_length=win_length),
                   center=False, return_complex=True)
    magnitude = torch.abs(S)
    phase = torch.exp(1j * torch.angle(S))
    return magnitude, phase


fs = 16000
farend_speech = "./farend_speech/farend_speech_fileid_9992.wav"
nearend_mic_signal = "./nearend_mic_signal/nearend_mic_fileid_9992.wav"
nearend_speech = "./nearend_speech/nearend_speech_fileid_9992.wav"
echo_signal = "./echo_signal/echo_fileid_9992.wav"

print("GPU是否可用：", torch.cuda.is_available())  # True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

farend_speech_magnitude, farend_speech_phase = spectrogram(farend_speech)  # 远端语音  振幅，相位
nearend_mic_magnitude, nearend_mic_phase = spectrogram(nearend_mic_signal)  # 近端麦克风语音 振幅，相位
nearend_speech_magnitude, nearend_speech_phase = spectrogram(nearend_speech)  # 近端语音振 幅，相位

farend_speech_magnitude = farend_speech_magnitude.to(device)
nearend_mic_phase = nearend_mic_phase.to(device)
nearend_mic_magnitude = nearend_mic_magnitude.to(device)

nearend_speech_magnitude = nearend_speech_magnitude.to(device)
nearend_speech_phase = nearend_speech_phase.to(device)

model = Base_model().to(device)  # 实例化模型
checkpoint = torch.load("../checkpoints/AEC_baseline/10.pth")
model.load_state_dict(checkpoint["model"])

X = torch.cat((farend_speech_magnitude, nearend_mic_magnitude), dim=0)
X = X.unsqueeze(0)
per_mask = model(X)  # [1, 322, 999]-->[1, 161, 999]

per_nearend_magnitude = per_mask * nearend_mic_magnitude  # 预测的近端语音 振幅

complex_stft = per_nearend_magnitude * nearend_mic_phase  # 振幅*相位=语音复数表示
print("complex_stft", complex_stft.shape)  # [1, 161, 999]

per_nearend = torch.istft(complex_stft, n_fft=320, hop_length=160, win_length=320,
                          window=torch.hann_window(window_length=320).to("cuda"))

torchaudio.save("./predict/nearend_speech_fileid_9992.wav", src=per_nearend.cpu().detach(), sample_rate=fs)
# print("近端语音", per_nearend.shape)    # [1, 159680]

y, _ = librosa.load(nearend_speech, sr=fs)
time_y = np.arange(0, len(y)) * (1.0 / fs)
recover_wav, _ = librosa.load("./predict/nearend_speech_fileid_9992.wav", sr=16000)
time_recover = np.arange(0, len(recover_wav)) * (1.0 / fs)

plt.figure(figsize=(8,6))
ax_1 = plt.subplot(3, 1, 1)
plt.title("近端语音和预测近端波形图", fontsize=14)
plt.plot(time_y, y, label="近端语音")
plt.plot(time_recover, recover_wav, label="深度学习生成的近端语音波形")
plt.xlabel('时间/s', fontsize=14)
plt.ylabel('幅值', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.subplots_adjust(top=0.932, bottom=0.085, left=0.110, right=0.998)
plt.subplots_adjust(hspace=0.809, wspace=0.365)  # 调整子图间距
plt.legend()

norm = matplotlib.colors.Normalize(vmin=-200, vmax=-40)
ax_2 = plt.subplot(3, 1, 2)
plt.title("近端语音频谱", fontsize=14)
plt.specgram(y, Fs=fs, scale_by_freq=True, sides='default', cmap="jet", norm=norm)
plt.xlabel('时间/s', fontsize=14)
plt.ylabel('频率/kHz', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.subplots_adjust(top=0.932, bottom=0.085, left=0.110, right=0.998)
plt.subplots_adjust(hspace=0.809, wspace=0.365)  # 调整子图间距

ax_3 = plt.subplot(3, 1, 3)
plt.title("深度学习生成的近端语音频谱", fontsize=14)
plt.specgram(recover_wav, Fs=fs, scale_by_freq=True, sides='default', cmap="jet", norm=norm)
plt.xlabel('时间/s', fontsize=14)
plt.ylabel('频率/kHz', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.subplots_adjust(top=0.932, bottom=0.085, left=0.110, right=0.998)
plt.subplots_adjust(hspace=0.809, wspace=0.365)  # 调整子图间距

def formatnum(x, pos):
    return '$%d$' % (x / 1000)


formatter = FuncFormatter(formatnum)
ax_2.yaxis.set_major_formatter(formatter)
ax_3.yaxis.set_major_formatter(formatter)


plt.show()

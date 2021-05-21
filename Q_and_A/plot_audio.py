# Author:凌逆战
# -*- coding:utf-8 -*-
"""
作用：
"""
import os

import librosa
import numpy as np
import matplotlib.pyplot as plt

rootpath = "../data_preparation/Synthetic/TRAIN"
echo_signal, _ = librosa.load(os.path.join(rootpath, "echo_signal", "echo_fileid_0.wav"), sr=16000)
farend_speech, _ = librosa.load(os.path.join(rootpath, "farend_speech", "farend_speech_fileid_0.wav"), sr=16000)
nearend_mic_signal, _ = librosa.load(os.path.join(rootpath, "nearend_mic_signal", "nearend_mic_fileid_0.wav"), sr=16000)
nearend_speech, _ = librosa.load(os.path.join(rootpath, "nearend_speech", "nearend_speech_fileid_0.wav"), sr=16000)

plt.subplot(4, 1, 1)
plt.plot(farend_speech)


plt.subplot(4, 1, 2)
plt.plot(echo_signal)

plt.subplot(4, 1, 3)
plt.plot(nearend_speech)

plt.subplot(4, 1, 4)

plt.plot(nearend_mic_signal)

plt.tight_layout()
plt.show()

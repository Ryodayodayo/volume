import matplotlib
matplotlib.use('TkAgg')

import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile



#wavファイル名を指定
filename = "青いベンチ ボーカル.wav"

#読み込み
fs, data = wavfile.read(filename)


#ステレオに対応
if data.ndim == 2:
    data = data[:,0] #左チャンネルのみ取り出す

#データをfloat32に変換
data = data.astype(np.float32) / 32768.0

#時間軸の作成
time = np.linspace(0, len(data) / fs, num = len(data))




# スレッショルドを0.5（最大振幅の50%）に設定
threshold = 0.3 * np.max(np.abs(data))

# 圧縮比率を設定（例えば4:1）
ratio = 4

#ソフトニー処理
def compressor_soft_knee(sample, threshold, ratio):
    abs_sample = np.abs(sample)
    if abs_sample < threshold:
        return sample
    else:
        # ソフトニー風のカーブ処理
        compressed = np.sign(sample) * (threshold + (abs_sample - threshold) / ratio)
        return compressed

# 音声データの圧縮
compressed_data = np.copy(data)

for i in range(len(data)):
    if np.abs(data[i]) > threshold:
        # 圧縮: 音量をレシオに基づいて圧縮
        compressed_data[i] = compressor_soft_knee(data[1], threshold, ratio)

#int16型に変更
compressed_data = np.clip(compressed_data, -1.0, 1.0) * 32767
compressed_data = compressed_data.astype(np.int16)

#最大の音量を見つける
"""max_amplitude = np.max(np.abs(compressed_data))

#正規化するための倍率処理
target_amplitude = 32767 * 0.5
scaling_factor =  target_amplitude / max_amplitude

#音量を正規化
normalized_data = compressed_data * scaling_factor
#normalized_data = np.tanh(normalized_data / 32767) * 32767 #簡単なリミッター処理"""
#compressed_data_int16 = np.int16(compressed_data)

#出力
wavfile.write('compressed_output.wav', fs, compressed_data)

#加工前と後の音声データを表示
plt.figure(figsize = (12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, data)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Original Waveform")

plt.subplot(2, 1, 2)
plt.plot(time, compressed_data)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Compressed Waveform")

plt.tight_layout()
plt.show()
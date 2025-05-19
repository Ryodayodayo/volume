import matplotlib
matplotlib.use('TkAgg')

import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf
import os

"""
つけたい機能

元の音源の音量からいい感じのスレッショルド、レシオ、kneeの値を自動決定する
音量レベルから自動でノーマライズする

スレッショルド、レシオ、kneeを画面上で設定(操作性も高くする)
ノーマライズの値を画面上で設定する
圧縮した音量を視覚化する
"""

#音声処理関数
def process_audio(input_path, output_path, threshold, ratio, target_level):
    #ファイル読み込み
    fs, data = wavfile.read(input_path)

    #ステレオに対応
    if data.ndim == 2:
        data = data[:,0] #左チャンネルのみ取り出す

    #データをfloat32に変換 
    data = data.astype(np.float32) / 32767.0   

    # コンプレッサー処理

    #スレッショルドを設定
    threshold = threshold * np.max(np.abs(data))

    def compressor_soft_knee(sample,threshold,ratio):
        abs_sample = np.abs(sample)
        if abs_sample < threshold:
            return sample
        else:
            # ソフトニー風のカーブ処理
            compressed = np.sign(sample) * (threshold + (abs_sample - threshold) / ratio)
            return compressed
        

    compressed_data = np.array([compressor_soft_knee(s, threshold, ratio) for s in data], dtype=np.float32)

    """
    #peakスケーリングで音割れ防止
    peak = np.max(np.abs(compressed_data))
    if peak > 1.0:
        compressed_data = compressed_data / peak
    """

    #音をノーマライズ
    max_amp = np.max(np.abs(compressed_data))
    scaling_factor = target_level / max_amp
    normalized_data = compressed_data * scaling_factor

    #出力
    sf.write(output_path, normalized_data, samplerate=fs, subtype='FLOAT')
 
    #ファイル名を元にグラフファイル名を生成
    base_filename = os.path.basename(input_path)
    graph_filename = f"graph_{os.path.splitext(base_filename)[0]}.png"

    if not os.path.exists('static'):
        os.makedirs('static')  # staticディレクトリがない場合は作成


    # グラフの保存
    graph_path = os.path.join('static', graph_filename)

    #時間軸の作成
    time = np.linspace(0, len(data) / fs, num = len(data))

    #加工前と後の音声データを表示
    plt.figure(figsize = (12, 6))

    #オリジナル
    plt.subplot(2, 1, 1)
    plt.plot(time, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Original Waveform")

    #float出力
    plt.subplot(2, 1, 2)
    plt.plot(time, normalized_data)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Compressed Waveform")

    plt.tight_layout()
    #グラフをファイルとして保存
    plt.savefig(graph_path)

    return graph_filename

"""
#wavファイル名を指定
filename = "青いベンチ ボーカル.wav"

#読み込み
fs, data = wavfile.read(filename)


#ステレオに対応
if data.ndim == 2:
    data = data[:,0] #左チャンネルのみ取り出す

#データをfloat32に変換
data = data.astype(np.float32) / 32767.0

#時間軸の作成
time = np.linspace(0, len(data) / fs, num = len(data))




# スレッショルドを0.6（最大振幅の30%）に設定
threshold = 0.6 * np.max(np.abs(data))

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
compressed_data = np.array([compressor_soft_knee(s, threshold, ratio) for s in data], dtype=np.float32)

# tanhでクリップ代わりに滑らかに圧縮
#compressed_data = np.tanh(compressed_data)

#peakスケーリングで音割れ防止
peak = np.max(np.abs(compressed_data))
if peak > 1.0:
    compressed_data = compressed_data / peak

#音をノーマライズ
max_amp = np.max(np.abs(compressed_data))
target_level = 0.7 #ターゲット音量 0.0 ~ 1.0
scaling_factor = target_level / max_amp
normalized_data = compressed_data * scaling_factor

#float出力
sf.write('compressed_output_float.wav', normalized_data, samplerate=fs, subtype='FLOAT')

#16int出力
#compressed_data_int16 = (compressed_data * 32767.0).astype(np.int16)
#wavfile.write('compressed_output.wav', fs, compressed_data_int16)


#加工前と後の音声データを表示
plt.figure(figsize = (12, 6))

#オリジナル
plt.subplot(2, 1, 1)
plt.plot(time, data)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Original Waveform")

#float出力
plt.subplot(2, 1, 2)
plt.plot(time, normalized_data)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Compressed Waveform")

plt.tight_layout()
plt.show()

"""
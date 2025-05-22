import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf

def db_to_linear(db):
    """dB値からリニア振幅比に変換"""
    return 10 ** (db / 20)

def linear_to_db(linear):
    """リニア振幅比からdB値に変換"""
    return 20 * np.log10(np.maximum(linear, 1e-10))

def compressor_envelope(data, threshold_db, ratio, attack_ms, release_ms, fs, knee_db=0):
    """
    attack/releaseを持つソフトニーコンプレッサー処理

    data: 入力信号(float32, -1.0〜1.0)
    threshold_db: スレッショルド（dB）
    ratio: レシオ
    attack_ms: アタック時間（ミリ秒）
    release_ms: リリース時間（ミリ秒）
    fs: サンプリング周波数
    knee_db: ソフトニー幅（dB）
    """

    # 入力信号の絶対値をdBに変換
    abs_db = linear_to_db(np.abs(data))

    # ゲイン計算用の配列初期化
    gain_reduction_db = np.zeros_like(abs_db)

    # スレッショルド周辺のソフトニー処理
    knee_start = threshold_db - knee_db / 2
    knee_end = threshold_db + knee_db / 2

    for i, level_db in enumerate(abs_db):
        if level_db < knee_start:
            gain_reduction_db[i] = 0  # スレッショルド以下は無圧縮
        elif level_db > knee_end:
            gain_reduction_db[i] = threshold_db + (level_db - threshold_db) / ratio - level_db
        else:
            # ソフトニー内はスムーズに減衰量を補間
            # 圧縮量 = x として線形補間（2次曲線で滑らかにしても良い）
            x = (level_db - knee_start) / knee_db
            compressed_db = threshold_db + (level_db - threshold_db) / ratio
            gain_reduction_db[i] = (1 - x) * 0 + x * (compressed_db - level_db)

    # ゲインをリニアに戻す
    target_gain = db_to_linear(gain_reduction_db)

    # アタック・リリースの係数計算
    attack_coef = np.exp(-1.0 / (attack_ms * 0.001 * fs))
    release_coef = np.exp(-1.0 / (release_ms * 0.001 * fs))

    smoothed_gain = np.zeros_like(target_gain)
    smoothed_gain[0] = target_gain[0]

    for i in range(1, len(target_gain)):
        if target_gain[i] < smoothed_gain[i-1]:
            smoothed_gain[i] = attack_coef * smoothed_gain[i-1] + (1 - attack_coef) * target_gain[i]
        else:
            smoothed_gain[i] = release_coef * smoothed_gain[i-1] + (1 - release_coef) * target_gain[i]

    # 入力信号にスムーズ化したゲインをかけて圧縮完了
    output = data * smoothed_gain

    return output

def normalize_audio(data, target_level=-1.0):
    """ピークをtarget_level(dB)にノーマライズ"""
    peak = np.max(np.abs(data))
    target_linear = db_to_linear(target_level)
    if peak == 0:
        return data
    return data * (target_linear / peak)

def process_audio_advanced(input_path, output_path,
                           threshold_db, ratio,
                           attack_ms, release_ms,
                           knee_db, normalize_level_db):
    # 読み込み
    fs, data = wavfile.read(input_path)

    # ステレオ対応（左のみ）
    if data.ndim == 2:
        data = data[:, 0]

    # float32変換
    data = data.astype(np.float32) / 32767.0

    # コンプレッサー処理
    compressed = compressor_envelope(data, threshold_db, ratio, attack_ms, release_ms, fs, knee_db)

    # ノーマライズ
    normalized = normalize_audio(compressed, normalize_level_db)


    # 出力
    sf.write(output_path, normalized, samplerate=fs, subtype='FLOAT')

    # グラフ描画
    base_filename = os.path.basename(input_path)
    graph_filename = f"graph_{os.path.splitext(base_filename)[0]}.png"
    if not os.path.exists('static'):
        os.makedirs('static')
    graph_path = os.path.join('static', graph_filename)

    time = np.linspace(0, len(data) / fs, num=len(data))

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, data)
    plt.title("Original Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(time, normalized)
    plt.title("Compressed & Normalized Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return graph_filename
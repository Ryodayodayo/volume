import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf
import logging

# CHUNKサイズ（サンプル数）を設定
CHUNK_SIZE = 1024


#ファイル全体を一度読み込んで最大値を得る
def get_peak(input_path):
    peak = 0.0
    while True:
        with sf.SoundFile(input_path) as sf_in :
            chunk = sf_in.read(CHUNK_SIZE, dtype='float32')
            if len(chunk) == 0 :
                break
            #ステレオだった場合モノラル(左側だけ)に変換
            if chunk.ndim == 2 :
                data = chunk[:,0]
            else:
                data = chunk
            peak = max(peak, np.max(np.abs(data))) #保存されているpeakとchunk内の最大値を比較し、大きい方をpeakに保存する   
        return peak    
                    

def db_to_linear(db):
    """dB値からリニア振幅比に変換"""
    return 10 ** (db / 20)

def linear_to_db(linear):
    """リニア振幅比からdB値に変換"""
    return 20 * np.log10(np.maximum(linear, 1e-10))

def compressor_envelope(data, threshold_db, ratio, attack_ms, release_ms, fs, knee_db):
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

def normalize_audio(data, target_level):
    """ピークをtarget_level(dB)にノーマライズ"""
    peak = np.max(np.abs(data))
    target_linear = db_to_linear(target_level)
    if peak == 0:
        return data
    return data * (target_linear / peak)

def apply_compressor(threshold_db, ratio, attack_ms, release_ms, fs, knee_db):
    def processor(data):
        return compressor_envelope(data, threshold_db, ratio, attack_ms, release_ms, fs, knee_db)
    return processor

def apply_normalize(normalize_level_db):
    def processor(data):
        return normalize_audio(data, normalize_level_db)
    return processor

def apply_reverb(data, fs, decay, delay_ms, repeats, mix):
    delay_samples = int(fs * delay_ms / 1000)
    reverb = np.zeros_like(data)

    for i in range(1, repeats + 1):
        attenuated = (decay ** i) * np.pad(data, (delay_samples * i, 0), mode='constant')[:-delay_samples * i]
        reverb += attenuated

    # mixパラメータでドライとウェットをブレンド
    output = (1 - mix) * data + mix * reverb

    # クリップ防止（-1〜1の範囲に収める）
    output = np.clip(output, -1.0, 1.0)
    return output

def apply_delay(data, fs, delay_ms, feedback, mix):
    delay_samples = int(fs * delay_ms / 1000)
    output = np.zeros_like(data)
    
    for i in range(delay_samples, len(data)):
        output[i] = data[i] + feedback * output[i - delay_samples]
    
    # 原音と混ぜる
    result = (1 - mix) * data + mix * output
    result = np.clip(result, -1.0, 1.0)
    return result

def apply_processing_chain(data, steps):
    for step in steps:
        data = step(data)
    return data

def process_audio_advanced(input_path, output_path,
                           threshold_db, ratio,
                           attack_ms, release_ms,
                           knee_db, normalize_level_db):
    # 読み込み
    data,fs = sf.read(input_path)

    # ステレオ対応（左のみ）
    if data.ndim == 2:
        data = data[:, 0]

    # init16型を正規化(最大値32768で割る)
    #data = data.astype(np.float32) / 32768.0

    # コンプレッサー処理
    #compressed = compressor_envelope(data, threshold_db, ratio, attack_ms, release_ms, fs, knee_db)

    # ノーマライズ
    #normalized = normalize_audio(compressed, normalize_level_db)

    processing_steps = [
        apply_normalize(normalize_level_db),  # ← ノーマライズ先
        apply_compressor(threshold_db, ratio, attack_ms, release_ms, fs, knee_db),
        apply_compressor(threshold_db, ratio, attack_ms, release_ms, fs, knee_db),
        apply_normalize(normalize_level_db),
        lambda data: apply_reverb(data, fs, decay=0.4, delay_ms=60, repeats=4, mix=0.2),
        lambda data: apply_delay(data, fs, delay_ms=200, feedback=0.25, mix=0.4),
    ]

    processed = apply_processing_chain(data, processing_steps)

    # 出力
    sf.write(output_path, processed, samplerate=fs, subtype='FLOAT')

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
    plt.plot(time, processed)
    plt.title("Processed Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return graph_filename


#inst用の処理関数
def process_mastering_audio(input_path, output_path, normalize):
    # 音声読み込み（float32で統一）
    data,fs = sf.read(input_path)
    #data = data.astype(np.float32) / 32768.0  # 16bit整数からfloatへ変換

    # ノーマライズ (値はボーカルに適応されてるものと同じ)
    normalized = normalize_audio(data, normalize)

    # float形式で保存
    sf.write(output_path, normalized, samplerate=fs, subtype='FLOAT')


#mixの処理関数

def process_mix_audio(inst_path, vocal_path, output_path, vocal_ratio, inst_ratio, offset_ms):
    # 読み込み
    inst_data, inst_sr = sf.read(inst_path)
    vocal_data, vocal_sr = sf.read(vocal_path)

    logging.info(f"vocal_ratio = {vocal_ratio}")
    logging.info(f"inst_ratio = {inst_ratio}")
    logging.info(f"offset_ms = {offset_ms}")

    # サンプリングレートが違ったらエラー
    if inst_sr != vocal_sr:
        raise ValueError("Sampling rates of inst and vocal do not match.")
    
    # vocal_data をステレオに変換（モノラルなら）
    if vocal_data.ndim == 1:
        vocal_data = np.stack([vocal_data, vocal_data], axis=1)

    #オフセット値をmsからサンプリングレート基準に変換
    offset_s = int(vocal_sr * offset_ms / 1000)    

    # オフセット処理
    if offset_s > 0:
        # vocalを遅らせる → 前に無音を追加
        pad = np.zeros((offset_s, 2))  # 2チャンネルの無音
        vocal_data = np.concatenate([pad, vocal_data], axis=0)
    elif offset_s < 0:
        # vocalを早める → 前から削る
        offset_s = abs(offset_s)
        vocal_data = vocal_data[offset_s:] 

    # 長さを合わせる（短い方に揃える）
    min_len = min(len(inst_data), len(vocal_data))
    inst_data = inst_data[:min_len]
    vocal_data = vocal_data[:min_len]

    # 単純に足し合わせてミックス（音割れを防ぐために 0.5 倍）
    mix = 0.5 * (inst_data * inst_ratio + vocal_data * vocal_ratio)

    # 書き出し
    sf.write(output_path, mix, inst_sr)
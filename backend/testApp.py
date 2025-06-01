import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf
import logging
import tempfile #一時ファイル用
import shutil #ファイルコピー用

# CHUNKサイズ（サンプル数）を設定
CHUNK_SIZE = 1024


#ファイル全体を一度読み込んで最大値を得る
def get_peak(input_path):
    peak = 0.0
    with sf.SoundFile(input_path) as sf_in :
        
        while True:
            chunk = sf_in.read(CHUNK_SIZE, dtype='float32')
            if len(chunk) == 0 :
                break
            #ステレオだった場合モノラル(左側だけ)に変換
            if chunk.ndim == 2 :
                data = chunk[:,0]
            else:
                data = chunk
            peak = max(peak, np.max(np.abs(data))) #保存されているpeakとchunk内の最大値を比較し、大きい方をpeakに保存する   
    logging.info("peakを計測したよ")
    return peak    

def db_to_linear(db):
    """dB値からリニア振幅比に変換"""
    return 10 ** (db / 20)

def linear_to_db(linear):
    """リニア振幅比からdB値に変換"""
    return 20 * np.log10(np.maximum(linear, 1e-10))

def compressor_envelope(audio_path, threshold_db, ratio, attack_ms, release_ms, fs, knee_db):
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp: #一時ファイルを作成

        temp_path = tmp.name
        
    prev_smoothed_gain = 1.0
        
    with sf.SoundFile(audio_path) as sf_in,\
         sf.SoundFile(temp_path, mode='w', samplerate=sf_in.samplerate, channels=sf_in.channels, format=sf_in.format) as sf_out:
         
        while True :
            chunk = sf_in.read(CHUNK_SIZE, dtype='float32')
            if chunk.ndim == 2:
                chunk = chunk[:, 0]

            if len(chunk) == 0:
                break
            
            else :
                # 入力信号の絶対値をdBに変換
                abs_db = linear_to_db(np.abs(chunk))

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
                smoothed_gain[0] = prev_smoothed_gain # 前チャンクの最後のゲインを引き継ぐ

                for i in range(1, len(target_gain)):
                    if target_gain[i] < smoothed_gain[i-1]:
                        smoothed_gain[i] = attack_coef * smoothed_gain[i-1] + (1 - attack_coef) * target_gain[i]
                    else:
                        smoothed_gain[i] = release_coef * smoothed_gain[i-1] + (1 - release_coef) * target_gain[i]

                prev_smoothed_gain = smoothed_gain[-1]  # 今回のチャンクの最後を保存        

                # 入力信号にスムーズ化したゲインをかけて圧縮完了
                chunk_compressed = chunk * smoothed_gain

                sf_out.write(chunk_compressed)

    shutil.move(temp_path, audio_path)
    logging.info("コンプレッサー終了")

    return audio_path

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

"""   

def normalize_audio(audio_path, target_level):
    """ピークをtarget_level(dB)にノーマライズ"""
    """audio_pathのデータを読み込んで、tempファイルに処理後のデータを書き出し。そのあとtempファイルをaudio_pathに上書き"""

    logging.info("ピークを得ようとするよ")
    peak = get_peak(audio_path)
    target_linear = db_to_linear(target_level)

    if peak == 0:
        print("音声にピークがないので処理をスキップしました")
        return
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp: #一時ファイルを作成

        temp_path = tmp.name

    with sf.SoundFile(audio_path) as sf_in,\
         sf.SoundFile(temp_path, mode='w', samplerate=sf_in.samplerate, channels=sf_in.channels, format=sf_in.format) as sf_out:
         
        while True :
            chunk = sf_in.read(CHUNK_SIZE, dtype='float32')
            if len(chunk) == 0:
                break
                
            chunk_normalized = chunk * (target_linear / peak)  

            sf_out.write(chunk_normalized)  
    
    shutil.move (temp_path, audio_path) #一時ファイルの名前を元ファイルに置き換え(上書き)
    logging.info("ノーマライズ終了")

    return audio_path    


def apply_compressor(threshold_db, ratio, attack_ms, release_ms, fs, knee_db):
    def processor(audio_path):
        logging.info("コンプレッサーを開始するよ")
        return compressor_envelope(audio_path, threshold_db, ratio, attack_ms, release_ms, fs, knee_db)
    return processor

def apply_normalize(normalize_level_db):
    def processor(audio_path):
        logging.info("ノーマライズを開始するよ")
        return normalize_audio(audio_path, normalize_level_db)
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

def apply_processing_chain(audio_path, steps):
    for step in steps:
        audio_path = step(audio_path)

    return audio_path

def read_and_downsample(file_path, target_points):
    """
    ダウンサンプリング
    """
    data, sr = sf.read(file_path)
    
    # モノラル化
    if data.ndim == 2:
        data = data[:, 0]
    
    # ダウンサンプリング
    if len(data) > target_points:
        indices = np.linspace(0, len(data) - 1, target_points, dtype=int)
        data = data[indices]
    
    return data

def create_graph(input_path, processed_path, graph_path, target_points=10000) :
        # ファイル情報を取得
    with sf.SoundFile(input_path) as sf_in:
        total_frames = sf_in.frames
        samplerate = sf_in.samplerate
        
    # 大きなファイルの場合のみダウンサンプリング
    if total_frames > target_points * 2:
        print(f"大きなファイル ({total_frames:,} samples) をダウンサンプリングします...")
        
        # チャンク単位で読み込みながらダウンサンプリング
        original_data = read_and_downsample(input_path, target_points)
        processed_data = read_and_downsample(processed_path, target_points)
        
        # 時間軸もダウンサンプリング
        time_axis = np.linspace(0, total_frames / samplerate, len(original_data))
        
    else:
        # 小さなファイルはそのまま読み込み
        original_data, sr1 = sf.read(input_path)
        processed_data, sr2 = sf.read(processed_path)
        
        # モノラル化
        if original_data.ndim == 2:
            original_data = original_data[:, 0]
        if processed_data.ndim == 2:
            processed_data = processed_data[:, 0]
            
        time_axis = np.linspace(0, len(original_data) / samplerate, len(original_data))
    
    # グラフ作成
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, original_data, linewidth=0.5)
    plt.title(f"Original Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, processed_data, linewidth=0.5)
    plt.title(f"Processed Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(graph_path, dpi=100, bbox_inches='tight')
    plt.close()  # メモリ解放
    
    print(f"グラフを保存しました: {graph_path}")


def process_audio_advanced(input_path, output_path,
                           threshold_db, ratio,
                           attack_ms, release_ms,
                           knee_db, normalize_level_db):

    # init16型を正規化(最大値32768で割る)
    #data = data.astype(np.float32) / 32768.0

    # コンプレッサー処理
    #compressed = compressor_envelope(data, threshold_db, ratio, attack_ms, release_ms, fs, knee_db)

    # ノーマライズ
    #normalized = normalize_audio(compressed, normalize_level_db)
    logging.info("音声処理に入った")

    PROCESSING = "processing"
    os.makedirs(PROCESSING, exist_ok=True) #beforeのフォルダがなければ作る

    processing_path = os.path.join(PROCESSING,os.path.basename(input_path))

    """
    dst_dir = os.path.dirname(first_path)  # コピー先のディレクトリ

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)  # コピー先のディレクトリがなければ作る
    """    

    shutil.copyfile(input_path, processing_path) #元データをpricessing_pathにコピー


    # 元データ読み込み(サンプリングレート取得用)
    data,fs = sf.read(input_path)
 


    
    logging.info("処理プロセス"),
    processing_steps = [
        apply_normalize(normalize_level_db),  
        apply_compressor(threshold_db, ratio, attack_ms, release_ms, fs, knee_db),
        apply_compressor(threshold_db, ratio, attack_ms, release_ms, fs, knee_db),
        apply_normalize(normalize_level_db),
        #lambda data: apply_reverb(data, fs, decay=0.4, delay_ms=60, repeats=4, mix=0.2),
        #lambda data: apply_delay(data, fs, delay_ms=200, feedback=0.25, mix=0.4),
    ]
    """
    processed = apply_processing_chain(input_path, processing_steps)
    """
    processed_path = apply_processing_chain(processing_path, processing_steps)

    logging.info("処理終了")
    
    #処理後のデータ読み込み
    processed, fs_processed = sf.read(processed_path)

    logging.info("データ読み込み完了")

    if processed.ndim == 2:
        processed = processed[:, 0] #モノラル化

    sf.write(output_path, processed, samplerate=fs_processed, subtype='FLOAT')    

    logging.info("データ書き出し完了")



    # 出力
    #sf.write(output_path, processed, samplerate=fs, subtype='FLOAT')

    # グラフ描画
    base_filename = os.path.basename(input_path)
    graph_filename = f"graph_{os.path.splitext(base_filename)[0]}.png"
    if not os.path.exists('static'):
        os.makedirs('static')
    graph_path = os.path.join('static', graph_filename)

    logging.info("グラフパス作成完了")

    create_graph(
        input_path, 
        output_path, 
        graph_path,
        target_points=10000,  # 1万ポイントに削減(ダウンサンプリング)
    )

    """
    time = np.linspace(0, len(data) / fs, num=len(data))

    plt.figure(figsize=(6, 3))
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
    """
    
    return graph_filename

def normalize_inst_audio(data, target_level): #inst用のノーマライズ関数
    """ピークをtarget_level(dB)にノーマライズ"""
    peak = np.max(np.abs(data))
    target_linear = db_to_linear(target_level)
    if peak == 0:
        return data
    return data * (target_linear / peak)


#inst用の処理関数
def process_mastering_audio(input_path, output_path, normalize):
    # 音声読み込み（float32で統一）

    data,fs = sf.read(input_path)

    #data = data.astype(np.float32) / 32768.0  # 16bit整数からfloatへ変換

    # ノーマライズ (値はボーカルに適応されてるものと同じ)
    """
    normalized = normalize_inst_audio(data, normalize)
    """
    
    # float形式で保存
    sf.write(output_path, data, samplerate=fs, subtype='FLOAT')


#mixの処理関数
"""
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

"""    
def process_mix_audio(inst_path, vocal_path, output_path, vocal_ratio, inst_ratio, offset_ms):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp: #一時ファイルを作成

        temp_path = tmp.name

    with sf.SoundFile(vocal_path) as sf_vocal_in,\
         sf.SoundFile(inst_path) as sf_inst_in,\
         sf.SoundFile(temp_path, mode='w', samplerate=sf_vocal_in.samplerate, channels=2, format=sf_vocal_in.format) as sf_out:
        
         logging.info(f"vocal_ratio = {vocal_ratio}")
         logging.info(f"inst_ratio = {inst_ratio}")
         logging.info(f"offset_ms = {offset_ms}")

        # サンプリングレートが違ったらエラー
         if sf_inst_in.samplerate != sf_vocal_in.samplerate:
            raise ValueError("Sampling rates of inst and vocal do not match.") 

        #オフセット値をmsからサンプリングレート基準に変換
         offset_s = int(sf_vocal_in.samplerate * offset_ms / 1000)

         # オフセット処理用の変数
         vocal_pad_remaining = 0
         inst_pad_remaining = 0
            
         if offset_s > 0:
            # vocalを遅らせる → vocalに無音パディング追加
            vocal_pad_remaining = abs(offset_s)
         elif offset_s < 0:
            # instに無音パディング追加
            inst_pad_remaining = abs(offset_s)      
         
         while True :
            chunk_vocal = sf_vocal_in.read(CHUNK_SIZE, dtype='float32')
            chunk_inst = sf_inst_in.read(CHUNK_SIZE, dtype='float32')

            # vocal_data をステレオに変換（モノラルなら）
            if chunk_vocal.ndim == 1:
                    chunk_vocal = np.stack([chunk_vocal, chunk_vocal], axis=1)

            # inst をステレオに変換（モノラルなら）
            if chunk_inst.ndim == 1:
                chunk_inst = np.stack([chunk_inst, chunk_inst], axis=1)
                    
            if len(chunk_vocal) == 0 and len(chunk_inst) == 0:
                break
            
            # vocal のパディング処理
            if vocal_pad_remaining > 0:
                pad_amount = min(vocal_pad_remaining, CHUNK_SIZE)
                pad = np.zeros((pad_amount, 2), dtype='float32')
                if len(chunk_vocal) > 0:
                    chunk_vocal = np.concatenate([pad, chunk_vocal], axis=0)
                else:
                    chunk_vocal = pad
                vocal_pad_remaining -= pad_amount

            # inst のパディング処理
            if inst_pad_remaining > 0:
                pad_amount = min(inst_pad_remaining, CHUNK_SIZE)
                pad = np.zeros((pad_amount, 2), dtype='float32')
                if len(chunk_inst) > 0:
                    chunk_inst = np.concatenate([pad, chunk_inst], axis=0)
                else:
                    chunk_inst = pad
                inst_pad_remaining -= pad_amount

            # 長さを揃える（短い方に合わせる）
            min_len = min(len(chunk_vocal), len(chunk_inst))
            if min_len == 0:
                continue

            chunk_vocal = chunk_vocal[:min_len]
            chunk_inst = chunk_inst[:min_len]

            # ミックス処理
            mix_chunk = 0.5 * (chunk_inst * inst_ratio + chunk_vocal * vocal_ratio)

            # 出力ファイルに書き込み
            sf_out.write(mix_chunk)



    shutil.move (temp_path, output_path) #一時ファイルの名前を元ファイルに置き換え(上書き)
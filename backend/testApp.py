import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf
import logging
import tempfile #一時ファイル用
import shutil #ファイルコピー用

# CHUNKサイズ（サンプル数）を設定
CHUNK_SIZE = 1024*1024


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
    logging.info('peak = {}'.format(peak))
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
         sf.SoundFile(temp_path, mode='w', samplerate=sf_in.samplerate, channels=1, format=sf_in.format) as sf_out:
         
        while True :
            chunk = sf_in.read(CHUNK_SIZE, dtype='float32')
            if chunk.ndim == 2:
                chunk = chunk[:, 0] #モノラルに変換

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

def normalize_audio(audio_path, target_amplitude):
    """ピークをtarget_level(dB)にノーマライズ"""
    """audio_pathのデータを読み込んで、tempファイルに処理後のデータを書き出し。そのあとtempファイルをaudio_pathに上書き"""

    logging.info("ピークを得ようとするよ")
    peak = get_peak(audio_path)

    if peak == 0:
        print("音声にピークがないので処理をスキップしました")
        return
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp: #一時ファイルを作成

        temp_path = tmp.name

    with sf.SoundFile(audio_path) as sf_in,\
         sf.SoundFile(temp_path, mode='w', samplerate=sf_in.samplerate, channels=1, format=sf_in.format) as sf_out:
         
        while True :
            chunk = sf_in.read(CHUNK_SIZE, dtype='float32')
            if chunk.ndim == 2:
                chunk = chunk[:, 0]
                
            if len(chunk) == 0:
                break
                
            chunk_normalized = chunk * (target_amplitude / peak)  

            sf_out.write(chunk_normalized)  
    
    shutil.move (temp_path, audio_path) #一時ファイルの名前を元ファイルに置き換え(上書き)
    logging.info("ノーマライズ終了")

    return audio_path    

def reverb_audio(audio_path, fs, decay, delay_ms, repeats, mix):
    """
    チャンクベースのリバーブ処理（修正版）
    
    audio_path: 入力音声ファイルのパス
    fs: サンプリング周波数
    decay: 減衰率
    delay_ms: ディレイ時間（ミリ秒）
    repeats: リピート回数
    mix: ドライ/ウェット比（0.0-1.0）
    """
    delay_samples = int(fs * delay_ms / 1000)
    
    # 各リピートのための循環バッファを初期化
    buffers = []
    buffer_positions = []
    
    for i in range(1, repeats + 1):
        buffer_size = delay_samples * i
        buffers.append(np.zeros(buffer_size))
        buffer_positions.append(0)  # 書き込み位置を追跡
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_path = tmp.name
        
    with sf.SoundFile(audio_path) as sf_in, \
         sf.SoundFile(temp_path, mode='w', samplerate=sf_in.samplerate, 
                     channels=1, format=sf_in.format) as sf_out:
        
        while True:
            chunk = sf_in.read(CHUNK_SIZE, dtype='float32')
            if chunk.ndim == 2:
                chunk = chunk[:, 0]
            
            if len(chunk) == 0:
                break
            
            reverb_chunk = np.zeros_like(chunk)
            
            # 各リピートを処理
            for i in range(repeats):
                attenuation = decay ** (i + 1)
                buffer = buffers[i]
                pos = buffer_positions[i]
                buffer_size = len(buffer)
                
                # 遅延信号を取得（循環バッファから読み取り）
                delayed_signal = np.zeros(len(chunk))
                
                for j in range(len(chunk)):
                    # 読み取り位置（現在の書き込み位置から遅延サンプル分戻る）
                    read_pos = (pos + j) % buffer_size
                    delayed_signal[j] = buffer[read_pos]
                    
                    # 新しいデータを書き込み
                    buffer[read_pos] = chunk[j]
                
                # 書き込み位置を更新
                buffer_positions[i] = (pos + len(chunk)) % buffer_size
                
                # 減衰した遅延信号をリバーブに加算
                reverb_chunk += attenuation * delayed_signal
            
            # ドライとウェットをミックス
            output_chunk = (1 - mix) * chunk + mix * reverb_chunk
            
            # クリッピング防止
            output_chunk = np.clip(output_chunk, -1.0, 1.0)
            
            sf_out.write(output_chunk)
    
    shutil.move(temp_path, audio_path)
    logging.info("リバーブ処理終了")
    
    return audio_path



def delay_audio(audio_path, fs, delay_ms, feedback, mix) :
    """
    チャンクベースのディレイ処理
    
    audio_path: 入力音声ファイルのパス
    fs: サンプリング周波数
    delay_ms: ディレイ時間（ミリ秒）
    feedback: フィードバック量
    mix: ドライ/ウェット比（0.0-1.0）
    """
    delay_samples = int(fs * delay_ms / 1000)
    
    # ディレイバッファを初期化
    delay_buffer = np.zeros(delay_samples)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_path = tmp.name
        
    with sf.SoundFile(audio_path) as sf_in, \
         sf.SoundFile(temp_path, mode='w', samplerate=sf_in.samplerate, 
                     channels=1, format=sf_in.format) as sf_out:
        
        while True:
            chunk = sf_in.read(CHUNK_SIZE, dtype='float32')
            if chunk.ndim == 2:
                chunk = chunk[:, 0]
            
            if len(chunk) == 0:
                break
            
            output_chunk = np.zeros_like(chunk)
            
            # チャンク内の各サンプルを処理
            for i in range(len(chunk)):
                # 現在のサンプル + フィードバックされた遅延信号
                delayed_sample = delay_buffer[0]
                output_chunk[i] = chunk[i] + feedback * delayed_sample
                
                # バッファを更新（FIFOキュー）
                delay_buffer[:-1] = delay_buffer[1:]  # 左シフト
                delay_buffer[-1] = output_chunk[i]    # 新しい値を末尾に追加
            
            # ドライとウェットをミックス
            result_chunk = (1 - mix) * chunk + mix * output_chunk
            
            # クリッピング防止
            result_chunk = np.clip(result_chunk, -1.0, 1.0)
            
            sf_out.write(result_chunk)
    
    shutil.move(temp_path, audio_path)
    logging.info("ディレイ処理終了")
    
    return audio_path


def apply_compressor(threshold_db, ratio, attack_ms, release_ms, fs, knee_db):
    def processor(audio_path):
        logging.info("コンプレッサーを開始するよ")
        return compressor_envelope(audio_path, threshold_db, ratio, attack_ms, release_ms, fs, knee_db)
    return processor

def apply_normalize(normalize_level):
    def processor(audio_path):
        logging.info("ノーマライズを開始するよ")
        return normalize_audio(audio_path, normalize_level)
    return processor

def apply_reverb(fs, decay, delay_ms, repeats, mix):
    def processor(audio_path):
        logging.info("リバーブを開始するよ")
        return reverb_audio(audio_path, fs, decay, delay_ms, repeats, mix)
    return processor

def apply_delay(fs, delay_ms, feedback, mix):
    def processor(audio_path):
        logging.info("ディレイを開始するよ")
        return delay_audio(audio_path, fs, delay_ms, feedback, mix)
    return processor


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

def create_graph(input_path, processed_path, graph_path, target_points) :
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
                           knee_db, normalize_level):

    logging.info("音声処理に入った")

    PROCESSING = "processing"
    os.makedirs(PROCESSING, exist_ok=True) #beforeのフォルダがなければ作る

    processing_path = os.path.join(PROCESSING,os.path.basename(input_path))



    shutil.copyfile(input_path, processing_path) #元データをpricessing_pathにコピー


    # 元データ読み込み(サンプリングレート取得用)
    data,fs = sf.read(input_path)

    
    logging.info("処理プロセス"),
    processing_steps = [
        apply_normalize(normalize_level),  
        apply_compressor(threshold_db, ratio, attack_ms, release_ms, fs, knee_db),
        apply_compressor(threshold_db, ratio, attack_ms, release_ms, fs, knee_db),
        apply_normalize(normalize_level),
        #apply_reverb(fs, decay=1, delay_ms=1, repeats=2, mix=0.2),
        apply_reverb(fs, decay=0.3, delay_ms=50, repeats=3, mix=0.3),
        #apply_delay(fs, delay_ms=200, feedback=0.25, mix=0.4),
    ]

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
        target_points=5000,  # 5000ポイントに削減(ダウンサンプリング)
    )
    
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



def process_mix_audio(inst_path, vocal_path, output_path, vocal_ratio, inst_ratio, offset_ms):
    """
    音声ファイルをミックスする関数
    
    Args:
        inst_path: インストゥルメンタル音声ファイルのパス
        vocal_path: ボーカル音声ファイルのパス
        output_path: 出力ファイルのパス
        vocal_ratio: ボーカルの音量比率
        inst_ratio: インストゥルメンタルの音量比率
        offset_ms: オフセット（ミリ秒）正の値でボーカルを遅らせ、負の値でインストを遅らせる
    """
    temp_path = None
    try:
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name

        # 音声ファイルを開く
        with sf.SoundFile(vocal_path) as sf_vocal_in, \
             sf.SoundFile(inst_path) as sf_inst_in:
            
            logging.info(f"vocal_path = {vocal_path}")
            logging.info(f"inst_path = {inst_path}")
            logging.info(f"output_path = {output_path}")
            logging.info(f"vocal_ratio = {vocal_ratio}")
            logging.info(f"inst_ratio = {inst_ratio}")
            logging.info(f"offset_ms = {offset_ms}")

            # サンプリングレートの確認
            if sf_inst_in.samplerate != sf_vocal_in.samplerate:
                raise ValueError("vocalとinstのサンプリングレートが違います")

            # 出力ファイルを開く
            with sf.SoundFile(temp_path, mode='w', 
                            samplerate=sf_vocal_in.samplerate, 
                            channels=2, 
                            format=sf_vocal_in.format) as sf_out:
                
                # オフセット値をサンプル数に変換
                offset_samples = int(sf_vocal_in.samplerate * offset_ms / 1000)
                
                # 遅延サンプル数を初期化
                vocal_delay_remaining = max(0, offset_samples)
                inst_delay_remaining = max(0, -offset_samples)
                
                # ファイル終了フラグ
                vocal_ended = False
                inst_ended = False
                
                logging.info(f"オフセットサンプル数: {offset_samples}")
                logging.info(f"ボーカル遅延: {vocal_delay_remaining}, インスト遅延: {inst_delay_remaining}")
                
                while True:
                    # ボーカルチャンクの処理
                    vocal_chunk = process_audio_chunk(
                        sf_vocal_in, vocal_delay_remaining, vocal_ended
                    )
                    vocal_data, vocal_delay_remaining, vocal_ended = vocal_chunk
                    
                    # インストチャンクの処理
                    inst_chunk = process_audio_chunk(
                        sf_inst_in, inst_delay_remaining, inst_ended
                    )
                    inst_data, inst_delay_remaining, inst_ended = inst_chunk
                    
                    # 終了条件の確認
                    if vocal_ended and inst_ended : #and vocal_delay_remaining == 0 and inst_delay_remaining == 0:
                        # 両方のチャンクが無音かどうか確認
                        if not np.any(vocal_data) and not np.any(inst_data):
                            logging.info("処理完了：両方の音声ストリームが終了")
                            break
                    
                    # ミックス処理
                    mixed_chunk = mix_audio_chunks(vocal_data, inst_data, vocal_ratio, inst_ratio)
                    
                    # 出力ファイルに書き込み
                    if np.any(mixed_chunk):
                        sf_out.write(mixed_chunk)
        
        # 一時ファイルを最終出力ファイルに移動
        shutil.move(temp_path, output_path)
        logging.info(f"音声ミックス完了: {output_path}")
        
    except Exception as e:
        logging.error(f"音声処理中にエラーが発生: {e}")
        # 一時ファイルのクリーンアップ
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def process_audio_chunk(sf_file, delay_remaining, file_ended):
    """
    音声チャンクを処理する関数
    
    Args:
        sf_file: SoundFileオブジェクト
        delay_remaining: 残りの遅延サンプル数
        file_ended: ファイル終了フラグ
    
    Returns:
        tuple: (音声データ, 更新された遅延サンプル数, 更新されたファイル終了フラグ)
    """
    # チャンクを無音で初期化
    chunk = np.zeros((CHUNK_SIZE, 2), dtype='float32')
    
    # 遅延処理
    if delay_remaining > 0:
        padding_size = min(delay_remaining, CHUNK_SIZE)
        delay_remaining -= padding_size
        frames_to_read = CHUNK_SIZE - padding_size
        start_index = padding_size
    else:
        frames_to_read = CHUNK_SIZE
        start_index = 0
    
    # ファイルからデータを読み込み
    if frames_to_read > 0 and not file_ended:

        try:
            audio_data = sf_file.read(frames_to_read, dtype='float32')
            
            if len(audio_data) == 0:
                file_ended = True
            else:
                # モノラルをステレオに変換
                if audio_data.ndim == 1:
                    audio_data = np.stack([audio_data, audio_data], axis=1)
                
                # 安全な範囲でデータをコピー
                end_index = min(start_index + len(audio_data), CHUNK_SIZE)
                actual_length = end_index - start_index
                
                if actual_length > 0:
                    chunk[start_index:end_index] = audio_data[:actual_length]
                
                # ファイル終端の確認
                if len(audio_data) < frames_to_read:
                    file_ended = True
                    
        except Exception as e:
            logging.error(f"音声データ読み込みエラー: {e}")
            file_ended = True
    
    return chunk, delay_remaining, file_ended


def mix_audio_chunks(vocal_chunk, inst_chunk, vocal_ratio, inst_ratio):
    """
    音声チャンクをミックスする関数
    
    Args:
        vocal_chunk: ボーカル音声チャンク
        inst_chunk: インストゥルメンタル音声チャンク
        vocal_ratio: ボーカルの音量比率
        inst_ratio: インストゥルメンタルの音量比率
    
    Returns:
        numpy.ndarray: ミックスされた音声チャンク
    """
    # クリッピング防止のため0.5を掛ける
    mixed = 0.5 * (vocal_chunk * vocal_ratio + inst_chunk * inst_ratio)
    
    # クリッピング防止（-1.0 ~ 1.0の範囲に制限）
    mixed = np.clip(mixed, -1.0, 1.0)
    
    return mixed 
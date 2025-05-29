import numpy as np
import soundfile as sf

fs = 44100  # サンプリング周波数
duration_sec = 5  # 5秒間

t = np.linspace(0, duration_sec, int(fs*duration_sec), endpoint=False)

# 1秒ごとに音量を変化させる包絡線（1→0.3→1→0.1→0.7など）
envelope = np.piecewise(t,
                        [t < 1, (t >= 1) & (t < 2), (t >= 2) & (t < 3), (t >= 3) & (t < 4), t >= 4],
                        [1.0, 0.3, 1.0, 0.1, 0.7])

# サイン波 440Hz（ラの音）
signal = 0.5 * np.sin(2 * np.pi * 440 * t)

# ノイズを少し足す
noise = 0.05 * np.random.randn(len(t))

# 信号に包絡線をかける
output = envelope * (signal + noise)

# float32で保存
sf.write("test_input.wav", output.astype(np.float32), fs)

print("test_input.wav を生成しました。")

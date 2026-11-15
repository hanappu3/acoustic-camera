"""
1-mic capture viewer (STM32H7 DFSDM -> USB virtual COM port -> PC)
マイク1個の取り込み表示（STM32H7のDFSDM → USB仮想COMポート → PC）

What this script does / このスクリプトがすること
  1. Wait for the board to send one capture (press the blue USER button).
     ボードから1回分のデータが届くのを待つ（青いUSERボタンを押す）。
  2. Remove the DC offset and plot the waveform and its spectrum.
     直流分を引き、波形と周波数スペクトルを表示する。
  3. Print the strongest frequency and its level in dBFS
     (check: a 1 kHz tone -> about 1000 Hz).
     一番強い周波数とその大きさ（dBFS）を表示する（確認：1 kHzの音なら約1000 Hz）。

  Re-analysis: set CSV_FILE to a saved file to redraw without the board.
  解析し直し：CSV_FILEに保存したファイル名を入れると、ボードなしで描き直せる。

Settings that must match the firmware / ファームウェアと合わせる設定
  CKOUT 3.2 MHz (64 MHz / 20), Sinc3, FOSR 64  ->  fs = 50 kHz
  115200 bps, 5000 samples

Part of: https://github.com/hanappu3/acoustic-camera
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import serial

plt.rcParams["font.family"] = "Meiryo"

# =============================================================================
# Settings / 設定
# =============================================================================
PORT = "COM3"        # change to your ST-LINK virtual COM port / デバイスマネージャーで確認したCOM番号に変える
BAUD = 115_200
FS = 50_000          # sampling rate [Hz] = 3.2 MHz / 64 / サンプリング周波数
FULL_SCALE = 64 ** 3 # Sinc3 with FOSR 64 -> max output 262144 = 0 dBFS / Sinc3・FOSR64の最大値
SKIP_MS = 5          # discard the first 5 ms (filter/mic settling) / 最初の5 msは捨てる
ZOOM_KHZ = 3         # upper limit of the zoomed spectrum [kHz] / 拡大図の上限
CSV_FILE = None      # set a saved CSV name (in results/) to re-analyze / 解析し直すときは results 内のCSVファイル名を入れる

# Save results next to this script, regardless of the terminal's current folder.
# ターミナルがどのフォルダにいても、このスクリプトと同じ場所の results フォルダに保存する
OUT_DIR = Path(__file__).resolve().parent.parent / "results"   # repo root / リポジトリ直下の results
OUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Receive one capture / 1回分を受け取る
# =============================================================================
def receive_capture(port, baud):
    """Read lines until START ... END and return the numbers between them.
    START と END の間の数値を読み取って返す"""
    with serial.Serial(port, baud, timeout=30) as ser:
        print(f"{port} を開きました。ボードの青いUSERボタンを押してください（30秒待ちます）")
        samples = []
        started = False
        while True:
            raw = ser.readline()
            if not raw:
                sys.exit("30秒待っても何も届きませんでした。COM番号と書き込みを確認してください")
            line = raw.decode(errors="ignore").strip()
            if line.startswith("ERROR"):
                sys.exit(f"ボードからエラー：{line}（DFSDMから値が来ていません。配線とクロック設定を確認）")
            if line == "START":
                started = True
                print("受信中...（約4秒）")
            elif line == "END":
                break
            elif started and line:
                samples.append(int(line))
    return np.array(samples, dtype=float)



# =============================================================================
# Get data / データを用意する
# =============================================================================
if CSV_FILE:
    x_raw = np.loadtxt(OUT_DIR / CSV_FILE)
    stamp = CSV_FILE.replace("mic1_capture_", "").replace(".csv", "") + "_reanalyzed"
    print(f"{CSV_FILE} を読み込みました")
else:
    x_raw = receive_capture(PORT, BAUD)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savetxt(OUT_DIR / f"mic1_capture_{stamp}.csv", x_raw, fmt="%d")

print(f"サンプル数：{len(x_raw)}")
print(f"生の値：最小 {x_raw.min():.0f}、最大 {x_raw.max():.0f}、平均（直流分）{x_raw.mean():.0f}")

# =============================================================================
# Analyze / 解析
# =============================================================================
skip = int(FS * SKIP_MS / 1000)
x = x_raw[skip:]                                   # drop the settling part / 落ち着くまでの部分を捨てる
x_ac = x - x.mean()                                # remove DC offset / 直流分を引く
t_ms = (np.arange(len(x)) + skip) / FS * 1000

# Amplitude spectrum scaled so that a sine of amplitude A shows A.
# 振幅Aの正弦波が、スペクトル上でAの高さになるように換算する
window = np.hanning(len(x_ac))
spec = np.abs(np.fft.rfft(x_ac * window)) * 2 / np.sum(window)
freqs = np.fft.rfftfreq(len(x_ac), 1 / FS)
spec_dbfs = 20 * np.log10(spec / FULL_SCALE + 1e-12)

i_peak = np.argmax(spec[1:]) + 1                   # skip the 0 Hz bin / 0 Hzは除く
peak_hz, peak_db = freqs[i_peak], spec_dbfs[i_peak]
print(f"一番強い周波数：{peak_hz:.0f} Hz、大きさ {peak_db:.1f} dBFS（振幅 約{spec[i_peak]:.0f}）")
print(f"周波数の分解能：{FS / len(x_ac):.1f} Hz")

# =============================================================================
# Plot / グラフ
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(9, 10))

ax = axes[0]
ax.plot(t_ms, x_ac, lw=0.8)
ax.set_xlabel("時間 [ms]")
ax.set_ylabel("値（直流分を除く）")
ax.set_title(f"マイク1個の波形（最初の{SKIP_MS} msを除く、{FS / 1000:.0f} kHz）")
ax.grid(alpha=0.3)

for ax, xmax, title in [(axes[1], FS / 2 / 1000, "周波数スペクトル（全体）"),
                        (axes[2], ZOOM_KHZ, f"周波数スペクトル（0〜{ZOOM_KHZ} kHzを拡大）")]:
    ax.plot(freqs / 1000, spec_dbfs, lw=0.8)
    ax.plot(peak_hz / 1000, peak_db, "o", color="tab:red", ms=6)
    ax.annotate(f"{peak_hz:.0f} Hz\n{peak_db:.1f} dBFS", (peak_hz / 1000, peak_db),
                textcoords="offset points", xytext=(10, -5), color="tab:red", fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-160, 0)
    ax.set_xlabel("周波数 [kHz]")
    ax.set_ylabel("大きさ [dBFS]")
    ax.set_title(title)
    ax.grid(alpha=0.3)

fig.tight_layout()
png_path = OUT_DIR / f"mic1_capture_{stamp}.png"
fig.savefig(png_path, dpi=150)
print(f"保存しました：{png_path}" + ("" if CSV_FILE else "（同じ名前の .csv も）"))
plt.show()

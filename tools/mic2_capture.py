"""
2-mic simultaneous capture viewer (STM32H7 DFSDM -> USB virtual COM port -> PC)
マイク2個の同時取り込み表示（STM32H7のDFSDM → USB仮想COMポート → PC）

What this script does / このスクリプトがすること
  1. Receive one capture of two channels (press the blue USER button).
     2ch分の取り込みを1回受け取る（青いUSERボタンを押す）。
  2. Overlay the two waveforms and compare their spectra and levels.
     2本の波形を重ね、スペクトルと大きさを比べる。
  3. Cross-correlate them to find the time difference (lag) and the side of the source.
     相互相関で時間差（ずれ）を求め、音源がどちら側かを判定する。

Lag sign / ずれの向き
  lag > 0 : sound reached mic A first (source on A's side)  / マイクAに先に届いた（A側）
  lag < 0 : sound reached mic B first (source on B's side)  / マイクBに先に届いた（B側）

Settings that must match the firmware / ファームウェアと合わせる設定
  CKOUT 3.2 MHz, Sinc3, FOSR 64 -> fs = 50 kHz, 115200 bps, 5000 samples x 2 ch

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
PORT = "COM3"
BAUD = 115_200
FS = 50_000            # sampling rate [Hz] / サンプリング周波数
C = 343.0              # speed of sound [m/s] / 音速
MIC_SPACING_M = 0.051  # distance between mic A and B [m] (side by side = 16 mm) / マイク間隔（隣どうし＝16 mm）
FULL_SCALE = 64 ** 3   # Sinc3, FOSR 64 -> 0 dBFS / フルスケール
CSV_FILE = None        # set a saved CSV name (in results/) to re-analyze / 解析し直すときはresults内のCSV名

OUT_DIR = Path(__file__).resolve().parent.parent / "results"   # repo root / リポジトリ直下のresults
OUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Receive / 受信
# =============================================================================
def receive_capture(port, baud):
    """Read "a,b" lines between START and END. / STARTとENDの間の「a,b」の行を読む"""
    with serial.Serial(port, baud, timeout=30) as ser:
        ser.reset_input_buffer()            # drop old data / 前に溜まったデータを捨てる
        print(f"{port} を開きました。ボードの青いUSERボタンを押してください（30秒待ちます）")
        rows, started, one_value = [], False, 0
        while True:
            raw = ser.readline()
            if not raw:
                sys.exit("30秒待っても何も届きませんでした。COM番号と書き込みを確認してください")
            line = raw.decode(errors="ignore").strip()
            if line.startswith("ERROR"):
                sys.exit(f"ボードからエラー：{line}")
            if line == "START":
                started = True
                print("受信中...（約7秒）")
            elif line == "END":
                if started:
                    break
                print("STARTより先にENDが届きました（送信の途中からつながった）。もう一度ボタンを押してください")
            elif started and "," in line:
                a, b = line.split(",")
                rows.append((int(a), int(b)))
            elif started and line:
                one_value += 1              # a line without a comma / カンマのない行
    if not rows:
        if one_value:
            sys.exit(f"数値が1個だけの行が{one_value}行届きました。ボードに1ch版のプログラムが入っています。h7_dfsdm_2chを書き込んでください")
        sys.exit("2chのデータが届きませんでした")
    return np.array(rows, dtype=float)


def xcorr_lag(a, b, max_lag):
    """score(lag) = sum a[n] * b[n + lag]. Peak at lag D means b is a delayed by D samples.
    相互相関。lag=Dで最大なら、BはAよりDサンプル遅れている"""
    n = len(a)
    lags = np.arange(-max_lag, max_lag + 1)
    core_a = a[max_lag : n - max_lag]
    scores = np.array([np.sum(core_a * b[max_lag + k : n - max_lag + k]) for k in lags])
    norm = np.sqrt(np.sum(core_a ** 2) * np.sum(b[max_lag : n - max_lag] ** 2))
    return lags, scores / norm          # normalized: 1.0 = identical shape / 1.0＝完全に同じ形


# =============================================================================
# Get data / データを用意
# =============================================================================
if CSV_FILE:
    data = np.loadtxt(OUT_DIR / CSV_FILE, delimiter=",")
    stamp = CSV_FILE.replace("mic2_capture_", "").replace(".csv", "") + "_reanalyzed"
else:
    data = receive_capture(PORT, BAUD)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savetxt(OUT_DIR / f"mic2_capture_{stamp}.csv", data, fmt="%d", delimiter=",")

a = data[:, 0] - data[:, 0].mean()      # mic A (Channel 0) / マイクA
b = data[:, 1] - data[:, 1].mean()      # mic B (Channel 1) / マイクB
print(f"サンプル数：{len(a)} × 2ch")

# =============================================================================
# Levels and spectra / 大きさとスペクトル
# =============================================================================
rms_a, rms_b = np.sqrt(np.mean(a ** 2)), np.sqrt(np.mean(b ** 2))
print(f"大きさ（RMS）：A {rms_a:.0f}、B {rms_b:.0f}、差 {20 * np.log10(rms_b / rms_a):+.1f} dB")

win = np.hanning(len(a))
freqs = np.fft.rfftfreq(len(a), 1 / FS)
spec = {k: np.abs(np.fft.rfft(v * win)) * 2 / np.sum(win) for k, v in (("A", a), ("B", b))}
for k in spec:
    i = np.argmax(spec[k][1:]) + 1
    print(f"マイク{k}：一番強い周波数 {freqs[i]:.0f} Hz、{20 * np.log10(spec[k][i] / FULL_SCALE):.1f} dBFS")

# =============================================================================
# Time difference / 時間差
# =============================================================================
max_lag = int(np.ceil(MIC_SPACING_M / C * FS)) + 2
lags, r = xcorr_lag(a, b, max_lag)
best = lags[np.argmax(r)]
print(f"相互相関：ずれ {best:+d} サンプル（{best / FS * 1e6:+.0f} µs）、一致度 {r.max():.3f}")
print(f"  探した範囲：±{max_lag} サンプル（マイク間隔 {MIC_SPACING_M * 1000:.0f} mm）")
if best > 0:
    side = "マイクA側"
elif best < 0:
    side = "マイクB側"
else:
    side = "正面（同時）"
sin_t = np.clip(-C * best / (FS * MIC_SPACING_M), -1, 1)   # A at -d/2, B at +d/2 / Aが左、Bが右
print(f"  判定：{side}（推定角 {np.rad2deg(np.arcsin(sin_t)):+.0f}°、B側がプラス）")

# =============================================================================
# Plot / グラフ
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(9, 10))
t_ms = np.arange(len(a)) / FS * 1000
show = t_ms < 3                                          # first 3 ms / 最初の3 ms
axes[0].plot(t_ms[show], a[show], label="マイクA（Ch0・立ち上がり）")
axes[0].plot(t_ms[show], b[show], label="マイクB（Ch1・立ち下がり）", alpha=0.8)
axes[0].set_xlabel("時間 [ms]")
axes[0].set_ylabel("値（直流分を除く）")
axes[0].set_title("2chの波形（最初の3 msを拡大）")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

for k in spec:
    axes[1].plot(freqs / 1000, 20 * np.log10(spec[k] / FULL_SCALE + 1e-12), lw=0.8, label=f"マイク{k}")
axes[1].set_xlim(0, FS / 2 / 1000)
axes[1].set_ylim(-160, 0)
axes[1].set_xlabel("周波数 [kHz]")
axes[1].set_ylabel("大きさ [dBFS]")
axes[1].set_title("2chのスペクトル")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

axes[2].plot(lags, r, ".-")
axes[2].axvline(best, color="tab:red", ls="--", label=f"最大：{best:+d} サンプル（{side}）")
axes[2].set_xlabel("ずれ [サンプル]（プラス＝Aに先に届いた）")
axes[2].set_ylabel("相互相関（1＝同じ形）")
axes[2].set_title("相互相関")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)

fig.tight_layout()
png_path = OUT_DIR / f"mic2_capture_{stamp}.png"
fig.savefig(png_path, dpi=150)
print(f"保存しました：{png_path}" + ("" if CSV_FILE else "（同じ名前の .csv も）"))
plt.show()

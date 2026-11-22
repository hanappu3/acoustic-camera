"""
4-mic simultaneous capture viewer (STM32H7 DFSDM -> USB virtual COM port -> PC)
マイク4個の同時取り込み表示（STM32H7のDFSDM → USB仮想COMポート → PC）

What this script does / このスクリプトがすること
  1. Receive one capture of four channels (press the blue USER button).
     4ch分の取り込みを1回受け取る（青いUSERボタンを押す）。
  2. Overlay the four waveforms and compare their levels and spectra.
     4本の波形を重ね、大きさとスペクトルを比べる。
  3. Cross-correlate mic B, C, D against mic A to check the time differences.
     マイクB・C・DをマイクAと相互相関し、時間差を確かめる。

Mic layout (straight line, equal pitch) / マイクの並び（一直線・等間隔）
  A (Ch0) - B (Ch1) - C (Ch2) - D (Ch3)

Lag sign / ずれの向き
  lag > 0 : sound reached mic A first (source on A's side) / マイクAに先に届いた（A側）
  Expected for a source on A's side: B, C, D lags grow as 1 : 2 : 3.
  A側の真横に音源があれば、B・C・Dのずれは 1：2：3 の比で大きくなる。

Settings that must match the firmware / ファームウェアと合わせる設定
  CKOUT 3.2 MHz, Sinc3, FOSR 64 -> fs = 50 kHz, 115200 bps, 2500 samples x 4 ch

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
FS = 50_000             # sampling rate [Hz] / サンプリング周波数
C = 343.0               # speed of sound [m/s] / 音速
MIC_PITCH_M = 0.01778   # distance between neighbors [m] (6 holes = 15.24 mm) / 隣どうしの間隔（6穴＝15.24 mm）
FULL_SCALE = 64 ** 3    # Sinc3, FOSR 64 -> 0 dBFS / フルスケール
CSV_FILE = None         # set a saved CSV name (in results/) to re-analyze / 解析し直すときはresults内のCSV名

NAMES = ["A", "B", "C", "D"]
OUT_DIR = Path(__file__).resolve().parent.parent / "results"   # repo root / リポジトリ直下のresults
OUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Receive / 受信
# =============================================================================
def receive_capture(port, baud):
    """Read "a,b,c,d" lines between START and END. / STARTとENDの間の「a,b,c,d」の行を読む"""
    with serial.Serial(port, baud, timeout=30) as ser:
        ser.reset_input_buffer()            # drop old data / 前に溜まったデータを捨てる
        print(f"{port} を開きました。ボードの青いUSERボタンを押してください（30秒待ちます）")
        rows, started, bad = [], False, 0
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
                print("STARTより先にENDが届きました。もう一度ボタンを押してください")
            elif started and line.count(",") == 3:
                rows.append([int(v) for v in line.split(",")])
            elif started and line:
                bad += 1                    # not a 4-value line / 4個の値でない行
    if not rows:
        sys.exit(f"4chのデータが届きませんでした（4個の値でない行：{bad}行）。h7_dfsdm_4chを書き込んだか確認してください")
    return np.array(rows, dtype=float)


def xcorr_lag(a, b, max_lag):
    """score(lag) = sum a[n] * b[n + lag]. Peak at lag D means b is a delayed by D samples.
    相互相関。lag=Dで最大なら、bはaよりDサンプル遅れている"""
    n = len(a)
    lags = np.arange(-max_lag, max_lag + 1)
    core_a = a[max_lag : n - max_lag]
    scores = np.array([np.sum(core_a * b[max_lag + k : n - max_lag + k]) for k in lags])
    norm = np.sqrt(np.sum(core_a ** 2) * np.sum(b[max_lag : n - max_lag] ** 2))
    return lags, scores / norm


def refine_peak(lags, r):
    """Fit a parabola through the peak and its two neighbors to get a sub-sample lag.
    山の頂点と両隣の3点に放物線を当てはめ、サンプルの間の位置を推定する"""
    i = int(np.argmax(r))
    if i == 0 or i == len(r) - 1:           # peak at the edge: cannot refine / 端なら補間しない
        return float(lags[i])
    y0, y1, y2 = r[i - 1], r[i], r[i + 1]
    return lags[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)


# =============================================================================
# Get data / データを用意
# =============================================================================
if CSV_FILE:
    data = np.loadtxt(OUT_DIR / CSV_FILE, delimiter=",")
    stamp = CSV_FILE.replace("mic4_capture_", "").replace(".csv", "") + "_reanalyzed"
else:
    data = receive_capture(PORT, BAUD)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savetxt(OUT_DIR / f"mic4_capture_{stamp}.csv", data, fmt="%d", delimiter=",")

print(f"サンプル数：{len(data)} × 4ch")
if np.any(data.std(axis=0) == 0):
    stuck = [NAMES[i] for i in range(4) if data[:, i].std() == 0]
    print(f"！ 値が変化していないマイク：{stuck}（DATA線・CLK線・電源を確認）")
x = data - data.mean(axis=0)                # remove DC per channel / チャネルごとに直流分を引く

# =============================================================================
# Levels and spectra / 大きさとスペクトル
# =============================================================================
rms = np.sqrt(np.mean(x ** 2, axis=0)) + 1e-12
win = np.hanning(len(x))
freqs = np.fft.rfftfreq(len(x), 1 / FS)
spec = np.abs(np.fft.rfft(x * win[:, None], axis=0)) * 2 / np.sum(win)
print("マイク  大きさ(RMS)  Aとの差   一番強い周波数")
for i, name in enumerate(NAMES):
    k = np.argmax(spec[1:, i]) + 1
    print(f"  {name}    {rms[i]:9.0f}   {20 * np.log10(rms[i] / rms[0]):+5.1f} dB   "
          f"{freqs[k]:6.0f} Hz（{20 * np.log10(spec[k, i] / FULL_SCALE + 1e-12):.1f} dBFS）")

# =============================================================================
# Time differences against mic A / マイクAとの時間差
# =============================================================================
max_lag = int(np.ceil(3 * MIC_PITCH_M / C * FS)) + 3
unit = MIC_PITCH_M / C * FS                 # max lag between neighbors / 隣どうしの最大のずれ
print(f"相互相関（基準：マイクA、探した範囲 ±{max_lag}、隣どうしの最大 {unit:.2f} サンプル）")
curves, peaks = [], []
for i in (1, 2, 3):
    lags, r = xcorr_lag(x[:, 0], x[:, i], max_lag)
    p = refine_peak(lags, r)
    curves.append(r)
    peaks.append(p)
    print(f"  A→{NAMES[i]}：ずれ {p:+.2f} サンプル、一致度 {r.max():.3f}（最大でも ±{unit * i:.2f}）")

# =============================================================================
# Plot / グラフ
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(9, 10))
t_ms = np.arange(len(x)) / FS * 1000
show = t_ms < 3
for i, name in enumerate(NAMES):
    axes[0].plot(t_ms[show], x[show, i], label=f"マイク{name}（Ch{i}）", alpha=0.85)
axes[0].set_xlabel("時間 [ms]")
axes[0].set_ylabel("値（直流分を除く）")
axes[0].set_title("4chの波形（最初の3 msを拡大）")
axes[0].legend(fontsize=9, ncol=4)
axes[0].grid(alpha=0.3)

for i, name in enumerate(NAMES):
    axes[1].plot(freqs / 1000, 20 * np.log10(spec[:, i] / FULL_SCALE + 1e-12), lw=0.7, label=f"マイク{name}")
axes[1].set_xlim(0, FS / 2 / 1000)
axes[1].set_ylim(-160, 0)
axes[1].set_xlabel("周波数 [kHz]")
axes[1].set_ylabel("大きさ [dBFS]")
axes[1].set_title("4chのスペクトル")
axes[1].legend(fontsize=9, ncol=4)
axes[1].grid(alpha=0.3)

for i, (r, p) in enumerate(zip(curves, peaks), start=1):
    line, = axes[2].plot(lags, r, ".-", label=f"A→{NAMES[i]}：{p:+.2f}")
    axes[2].axvline(p, color=line.get_color(), ls="--", lw=0.8)
axes[2].set_xlabel("ずれ [サンプル]（プラス＝Aに先に届いた）")
axes[2].set_ylabel("相互相関（1＝同じ形）")
axes[2].set_title("マイクAを基準にした相互相関")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)

fig.tight_layout()
png_path = OUT_DIR / f"mic4_capture_{stamp}.png"
fig.savefig(png_path, dpi=150)
print(f"保存しました：{png_path}" + ("" if CSV_FILE else "（同じ名前の .csv も）"))
plt.show()

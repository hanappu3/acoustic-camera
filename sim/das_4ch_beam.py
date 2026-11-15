"""
Stage 3: 4-mic linear array, delay-and-sum beamforming
Stage 3：マイク4個の直線配置と遅延和（方向ごとの音の強さを求める）

What this script does / このスクリプトがすること
  For every candidate direction, shift each mic's recording back by the delay
  expected from that direction and add them. If the guess is right, the waveforms
  line up and the sum is strong.
  方向を1°ずつ仮定し、その方向から来たときの遅れだけ各マイクの録音を戻して足す。
  仮定が当たっていれば波形がそろい、足した音が強くなる。

  Compare three source bands with 4 mics at 16 mm spacing (board width limit):
  マイク4個・間隔16 mm（基板幅で決まる最小間隔）で、音の帯域を3通り比べる。
    - wideband noise (up to 24 kHz) / 広帯域の雑音
    - noise up to 10 kHz            / 10 kHzまでの雑音
    - noise 0.5 - 2 kHz             / 0.5〜2 kHzの雑音（人の声や機械音に近い低い音）

Expected / 期待される結果
  One source at +30 deg  : peak at 30 deg in all cases, but much sharper for wideband
  Two sources at -30/+30 : wideband -> two peaks / 0.5-2 kHz -> one wrong peak at 0 deg
  音源1つ：どの帯域でも山は30°。ただし広帯域ほど山が鋭い。
  音源2つ：広帯域は山2つで正しい。低い音だけだと0°に山1つ（誤り）。

Part of: https://github.com/hanappu3/acoustic-camera
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

# =============================================================================
# Conditions / 条件
# =============================================================================
c, fs = 343.0, 48_000
M = 4                    # number of mics / マイクの数
d = 0.016                # spacing [m] (16 mm) / マイク間隔
R = 1.0                  # source distance [m] / 音源までの距離
N = 4096                 # recording length [samples] / 録音の長さ
SNR_DB = 20

mic_x = (np.arange(M) - (M - 1) / 2) * d     # x positions: -24, -8, +8, +24 mm / マイクのx座標
freqs = np.fft.rfftfreq(N, 1 / fs)          # frequency of each FFT bin / FFTの各成分の周波数
angles = np.arange(-90, 91, 1)              # candidate directions / 仮定する方向

BANDS = [(None, "広帯域の雑音（〜24 kHz）"),
         ((0, 10_000), "10 kHzまでの雑音"),
         ((500, 2_000), "0.5〜2 kHzの雑音（低い音）")]


# =============================================================================
# Functions / 関数
# =============================================================================
def delay(signal_spectrum, tau):
    """Delay a signal by tau seconds (any value, not only whole samples) in the frequency domain.
    信号を tau 秒遅らせる。周波数ごとに位相を回すので、サンプルの間の半端な遅れも作れる"""
    return signal_spectrum * np.exp(-2j * np.pi * freqs * tau)


def simulate(thetas_deg, band, seed=0):
    """Recordings of M mics for one or more noise sources.
    1つ以上の雑音源について、M個のマイクの録音を作る"""
    rng = np.random.default_rng(seed)
    recs = np.zeros((M, N))
    for theta_deg in thetas_deg:
        S = np.fft.rfft(rng.standard_normal(N))          # source spectrum / 音源の周波数成分
        if band is not None:
            S[(freqs < band[0]) | (freqs > band[1])] = 0  # keep only the band / 帯域外を消す
        th = np.deg2rad(theta_deg)
        src = np.array([R * np.sin(th), R * np.cos(th)])
        for m in range(M):
            L = np.linalg.norm(src - np.array([mic_x[m], 0.0]))   # exact distance / 正確な距離
            recs[m] += np.fft.irfft(delay(S, L / c), N)
    recs += rng.standard_normal(recs.shape) * recs.std() / 10 ** (SNR_DB / 20)
    return recs


def delay_and_sum(recs):
    """Power for each candidate direction, in dB (max = 0 dB).
    仮定した方向ごとの「足した音の強さ」をdBで返す（最大を0 dBとする）"""
    X = np.fft.rfft(recs, axis=1)
    power = []
    for a in angles:
        tau = mic_x * np.sin(np.deg2rad(a)) / c        # expected lead of each mic / 各マイクが早く届く時間
        Y = sum(delay(X[m], tau[m]) for m in range(M))  # shift back and add / 戻して足す
        power.append(np.sum(np.abs(Y) ** 2))
    power = np.array(power)
    return 10 * np.log10(power / power.max())


def peaks(p_db):
    """Angles of local maxima. / 山（周りより高い点）の角度"""
    return [int(angles[i]) for i in range(1, len(p_db) - 1) if p_db[i] > p_db[i - 1] and p_db[i] > p_db[i + 1]]


# =============================================================================
# Figure 1 & 2 / 図1と図2
# =============================================================================
for fig_no, (sources, title) in enumerate([([30], "音源1つ（+30°）"), ([-30, 30], "音源2つ（−30°と+30°）")], start=1):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for band, label in BANDS:
        p_db = delay_and_sum(simulate(sources, band))
        ax.plot(angles, p_db, label=label, lw=2)
        print(f"[{title}] {label}: peaks at {peaks(p_db)} deg")
    for s in sources:
        ax.axvline(s, color="gray", ls="--", lw=1)
    ax.set_xlabel("仮定した方向 [°]")
    ax.set_ylabel("足した音の強さ [dB]（最大＝0）")
    ax.set_title(f"マイク4個・間隔16 mm：{title}", fontsize=11)
    ax.set_ylim(-10, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"stage3_beam_{fig_no}.png", dpi=150)

plt.show()

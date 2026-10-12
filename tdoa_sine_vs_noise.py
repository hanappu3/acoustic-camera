"""
Stage 2: Why a pure tone breaks 2-mic TDOA (spatial aliasing)
Stage 2：正弦波（1つの周波数の音）だと2マイクの方向推定が間違える理由

What this script does / このスクリプトがすること
  Compare three sources -- broadband noise, 1 kHz sine, 3 kHz sine --
  using the same 2-mic cross-correlation as Stage 1.
  雑音・1 kHz正弦波・3 kHz正弦波の3種類の音で、Stage 1と同じ方法を比べる。

Key condition / 重要な条件
  A pure tone is unambiguous only if  d < c / (2 f)  (half wavelength).
  For d = 150 mm: f < 1143 Hz.
  正弦波で方向が一意に決まるのは、マイク間隔が波長の半分より狭いときだけ。

Expected / 期待される結果
  noise : max error 4.0 deg (same as Stage 1)
  1 kHz : max error 4.0 deg (below the limit, OK)
  3 kHz : about half of the angles are badly wrong (mirror peaks)

Part of: https://github.com/hanappu3/acoustic-camera
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

# =============================================================================
# Conditions / 条件
# =============================================================================
c, fs, d, R = 343.0, 48_000, 0.15, 1.0
N = 4000                 # recording length [samples] / 録音の長さ
SNR_DB = 20              # mic noise: signal is 10x larger than noise / マイク雑音（信号は雑音の10倍）
micA = np.array([-d / 2, 0.0])
micB = np.array([+d / 2, 0.0])
max_lag = int(np.ceil(d / c * fs)) + 2   # search range / 探すずれの範囲

SOURCES = [("noise", 0), ("sine", 1000), ("sine", 3000)] # (kind, frequency) / 音の種類と周波数


# =============================================================================
# Functions / 関数
# =============================================================================
def make_source(kind, f):
    """A long, continuously sounding source. / ずっと鳴っている長い音を作る"""
    n = np.arange(N + 800)
    if kind == "noise":
        return np.random.default_rng(0).standard_normal(len(n))
    return np.sin(2 * np.pi * f * n / fs)


def record(source, theta_deg, rng):
    """Cut out what each mic hears, then add mic noise.
    各マイクに届いた部分を切り出し、マイクの雑音を足す"""
    theta = np.deg2rad(theta_deg)
    src = np.array([R * np.sin(theta), R * np.cos(theta)])
    n_A = int(round(np.linalg.norm(src - micA) / c * fs))
    n_B = int(round(np.linalg.norm(src - micB) / c * fs))

    # A mic that is n samples farther hears the source n samples "older".
    # n サンプル遠いマイクには、n サンプル前の音が届いている
    recA = source[600 - n_A : 600 - n_A + N].copy()
    recB = source[600 - n_B : 600 - n_B + N].copy()

    noise_level = np.std(recA) / 10 ** (SNR_DB / 20)
    recA += rng.standard_normal(N) * noise_level
    recB += rng.standard_normal(N) * noise_level
    return recA, recB, n_A - n_B


def xcorr(recA, recB):
    """Cross-correlation, ignoring the edges affected by np.roll wrap-around.
    相互相関（np.rollで端が回り込む部分は使わない）"""
    lags = np.arange(-max_lag, max_lag + 1)
    core = slice(max_lag, N - max_lag)
    scores = np.array([np.sum(recA[core] * np.roll(recB, k)[core]) for k in lags])
    return lags, scores


def lag_to_angle(lag):
    """sin θ = c·lag / (fs·d). Clip to [-1, 1] because arcsin only accepts that range.
    ずれを角度に直す。arcsinは-1〜1しか受け付けないので範囲に収める"""
    return np.rad2deg(np.arcsin(np.clip(c * lag / (fs * d), -1, 1)))


def label(kind, f):
    return "雑音" if kind == "noise" else f"正弦波 {f} Hz"


rng = np.random.default_rng(1)

# =============================================================================
# Figure 1: cross-correlation at θ = 30° for each source
# 図1：θ=30°での相互相関を3種類の音で比べる
# =============================================================================
fig1, axes = plt.subplots(len(SOURCES), 1, figsize=(8, 8), sharex=True)
for ax, (kind, f) in zip(axes, SOURCES):
    recA, recB, true_lag = record(make_source(kind, f), 30, rng)
    lags, scores = xcorr(recA, recB)
    best = lags[np.argmax(scores)]
    ax.plot(lags, scores / scores.max(), ".-")
    ax.axvline(true_lag, color="tab:green", ls="--", label=f"本物のずれ {true_lag}")
    ax.axvline(best, color="tab:red", ls=":", lw=2, label=f"選ばれたずれ {best} → {lag_to_angle(best):.1f}°")
    ax.axvspan(-d / c * fs, d / c * fs, color="gray", alpha=0.1)
    ax.set_title(f"{label(kind, f)}：θ=30°", fontsize=11)
    ax.set_ylabel("相互相関（最大=1）")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    print(f"{label(kind, f)}: true lag = {true_lag}, chosen = {best} -> {lag_to_angle(best):.1f} deg")
axes[-1].set_xlabel("Bをずらした量 [サンプル]（灰色＝実際にありうる範囲）")
fig1.tight_layout()
fig1.savefig("stage2_xcorr_compare.png", dpi=150)

# =============================================================================
# Figure 2: angle sweep for each source
# 図2：-60°〜+60°の角度スイープを3種類の音で比べる
# =============================================================================
true_angles = np.arange(-60, 61, 1)
fig2, axes = plt.subplots(len(SOURCES), 1, figsize=(8, 9), sharex=True)
for ax, (kind, f) in zip(axes, SOURCES):
    source = make_source(kind, f)
    est = []
    for t in true_angles:
        recA, recB, _ = record(source, t, rng)
        lags, scores = xcorr(recA, recB)
        est.append(lag_to_angle(lags[np.argmax(scores)]))
    est = np.array(est)
    err = est - true_angles
    wrong = int(np.sum(np.abs(err) > 10))
    ax.plot(true_angles, true_angles, "--", color="gray")
    ax.plot(true_angles, est, ".", ms=4)
    ax.set_title(f"{label(kind, f)}：最大誤差 {np.max(np.abs(err)):.1f}°、10°以上外れた角度 {wrong}/121", fontsize=11)
    ax.set_ylabel("推定角度 [°]")
    ax.grid(alpha=0.3)
    print(f"{label(kind, f)}: max error = {np.max(np.abs(err)):.1f} deg, wrong(>10 deg) = {wrong}/121")
axes[-1].set_xlabel("本当の角度 [°]")
fig2.tight_layout()
fig2.savefig("stage2_sweep_compare.png", dpi=150)

print(f"half-wavelength limit for d = {d*1000:.0f} mm: f < {c / (2 * d):.0f} Hz / 正弦波で方向が一意に決まる上限")
plt.show()

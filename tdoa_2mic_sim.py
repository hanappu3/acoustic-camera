"""
2-microphone TDOA simulation
2マイクの到来時間差（TDOA）シミュレーション

What this script does / このスクリプトがすること
  1. Simulate a noise source and its arrival at two microphones (spherical wave).
     雑音の音源が、2つのマイクに円（球面）の波として届く様子を作る。
  2. Estimate the time difference of arrival by cross-correlation.
     相互相関で、2つの録音の時間差（ずれ）を求める。
  3. Convert the time difference back into the direction of arrival.
     ずれから音源の方向（到来角）を逆算する。

Default conditions / 既定の条件
  c = 343 m/s, fs = 48 kHz, d = 150 mm, theta = 30 deg, R = 1 m

Expected output / 期待される結果
  lag = 10 samples, estimated angle = 28.4 deg (true: 30 deg)
  The 1.6 deg error comes from integer-sample quantization
  (true delay is 10.47 samples; 1 sample = 7.1 mm of path difference).
  誤差1.6°は、ずれを整数サンプルで測ることによる（本当は10.47サンプル、1サンプル＝道のり7.1 mm）。

Part of: https://github.com/hanappu3/acoustic-camera
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"  # Japanese font on Windows / Windows用の日本語フォント

# =============================================================================
# Conditions / 条件
# =============================================================================
c = 343.0          # speed of sound [m/s] / 音速
fs = 48_000        # sampling rate [Hz] / サンプリング周波数
d = 0.15           # microphone spacing [m] / マイク間隔
theta_deg = 30     # source direction [deg], 0 = broadside, + = right / 音源の方向（正面0°、右がプラス）
R = 1.0            # source distance from array center [m] / アレイ中心から音源までの距離

# =============================================================================
# Geometry: origin at array center, y-axis = broadside
# 配置：マイクの中点を原点、正面をy方向とする
# =============================================================================
theta = np.deg2rad(theta_deg)
src = np.array([R * np.sin(theta), R * np.cos(theta)])  # source (x, y) / 音源の位置
micA = np.array([-d / 2, 0.0])                         # left mic / 左のマイク
micB = np.array([+d / 2, 0.0])                         # right mic / 右のマイク

# =============================================================================
# Step 1: Simulate the received signals
# 手順1：各マイクの録音を作る
# =============================================================================
# Exact source-to-mic distances (no far-field approximation).
# 音源から各マイクまでの正確な距離（三平方の定理。遠方界の近似は使わない）
L_A = np.linalg.norm(src - micA)
L_B = np.linalg.norm(src - micB)

# Propagation delay rounded to integer samples.
# 届くまでの時間を、整数のサンプル数に丸める
n_A = int(round(L_A / c * fs))
n_B = int(round(L_B / c * fs))

print(f"L_A = {L_A*1000:.1f} mm, L_B = {L_B*1000:.1f} mm")
print(f"n_A = {n_A}, n_B = {n_B}, true lag = {n_A - n_B} samples / ずれ（正解）")

# Broadband noise source (similar to an air leak hiss).
# 広帯域の雑音を音源にする（エア漏れの「シャー」という音に近い）
rng = np.random.default_rng(0)   # fixed seed for reproducibility / 毎回同じ乱数にする
N = 2000                         # source length [samples] / 音の長さ
source = rng.standard_normal(N)

# Each mic records the same source, delayed by its own propagation time.
# 各マイクの録音＝同じ音源を、届くまでのサンプル数だけ遅らせたもの
total = N + 300                  # buffer long enough for the delay / 遅れても収まる長さ
recA = np.zeros(total)
recB = np.zeros(total)
recA[n_A : n_A + N] = source
recB[n_B : n_B + N] = source

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(recA, label="マイクA", linewidth=1)
ax.plot(recB, label="マイクB", linewidth=1, alpha=0.7)
ax.set_xlim(n_B - 5, n_B + 60)   # zoom in on the onset / 届き始めを拡大
ax.set_xlabel("サンプル番号")
ax.set_ylabel("振幅")
ax.set_title(f"θ={theta_deg}°、距離{R} m：マイクBのほうが先に届く")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("tdoa_step1_signals.png", dpi=150)
plt.show()

# =============================================================================
# Step 2: Estimate the lag by cross-correlation
# 手順2：相互相関でずれを求める
# =============================================================================
# Slide recB over recA; at each lag, multiply sample-by-sample and sum.
# The sum peaks where the two waveforms line up.
# Bを少しずつずらしてAに重ね、同じ時刻どうし掛けて足す。形が重なる位置で最大になる。
max_lag = 30                     # search range [samples] / 調べる範囲
lags = np.arange(-max_lag, max_lag + 1)
scores = []

for lag in lags:
    shifted_B = np.roll(recB, lag)        # delay recB by `lag` samples / Bをlagだけ後ろにずらす
    scores.append(np.sum(recA * shifted_B))

scores = np.array(scores)
best_lag = lags[np.argmax(scores)]        # lag with the highest score / 最大になるずらし量

print(f"estimated lag = {best_lag} samples (true: {n_A - n_B}) / 相互相関で求めたずれ")

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(lags, scores, marker="o", markersize=3)
ax2.axvline(best_lag, color="tab:red", linestyle="--", label=f"最大：{best_lag} サンプル")
ax2.set_xlabel("Bをずらした量 [サンプル]")
ax2.set_ylabel("掛けて足した合計")
ax2.set_title("相互相関：形がぴったり重なるずらし量で最大になる")
ax2.legend()
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("tdoa_step2_xcorr.png", dpi=150)
plt.show()

# =============================================================================
# Step 3: Convert the lag to a direction of arrival (far-field model)
# 手順3：ずれから到来角を逆算する（遠方界の式：sinθ = c·τ / d）
# =============================================================================
sin_est = c * best_lag / (fs * d)
theta_est = np.rad2deg(np.arcsin(sin_est))
print(f"estimated angle = {theta_est:.1f} deg (true: {theta_deg} deg) / 推定した到来角")
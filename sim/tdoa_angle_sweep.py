"""
Angle sweep of 2-mic TDOA estimation
2マイクTDOAの角度スイープ：-60°〜+60°で推定誤差を調べる

Expected: the estimate is a staircase, max error about 4 deg.
期待される結果：推定は階段状になり、最大誤差は約4°。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

# Conditions / 条件
c, fs, d, R, N = 343.0, 48_000, 0.15, 1.0, 2000
rng = np.random.default_rng(0)
source = rng.standard_normal(N)        # noise source / 雑音の音源
micA = np.array([-d / 2, 0.0])
micB = np.array([+d / 2, 0.0])


def estimate_angle(theta_deg):
    """Simulate 2 mics and estimate the angle by cross-correlation.
    角度を入れると、相互相関で推定した角度を返す（昨日の手順1〜3をまとめたもの）"""
    # Step 1: received signals / 各マイクの録音を作る
    theta = np.deg2rad(theta_deg)
    src = np.array([R * np.sin(theta), R * np.cos(theta)])
    n_A = int(round(np.linalg.norm(src - micA) / c * fs))
    n_B = int(round(np.linalg.norm(src - micB) / c * fs))
    recA = np.zeros(N + 300)
    recB = np.zeros(N + 300)
    recA[n_A : n_A + N] = source
    recB[n_B : n_B + N] = source

    # Step 2: cross-correlation / 相互相関でずれを求める
    max_lag = int(np.ceil(d / c * fs)) + 2     # 真横のときの最大のずれ ＋ 余裕2
    lags = np.arange(-max_lag, max_lag + 1)
    scores = [np.sum(recA * np.roll(recB, lag)) for lag in lags]
    best_lag = lags[np.argmax(scores)]

    # Step 3: lag to angle / ずれから角度を逆算
    return np.rad2deg(np.arcsin(c * best_lag / (fs * d)))


# Sweep -60..+60 deg / 角度を1°ずつ動かして推定する
true_angles = np.arange(-60, 61, 1)
est_angles = np.array([estimate_angle(t) for t in true_angles])
error = est_angles - true_angles

print(f"max error = {np.max(np.abs(error)):.1f} deg / 最大誤差")
print(f"distinct estimates = {len(np.unique(np.round(est_angles, 3)))} / 推定値の種類の数（121通りの角度に対して）")

# Plot / グラフ
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
ax1.plot(true_angles, true_angles, "--", color="gray", label="理想（誤差ゼロ）")
ax1.plot(true_angles, est_angles, ".-", label="推定")
ax1.set_ylabel("推定した角度 [°]")
ax1.set_title(f"2マイク（間隔{d*1000:.0f} mm、{fs/1000:.0f} kHz）：推定は階段になる")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(true_angles, error, ".-", color="tab:red")
ax2.set_xlabel("本当の角度 [°]")
ax2.set_ylabel("誤差 [°]")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"tdoa_angle_sweep_d{d*1000:.0f}mm_fs{fs/1000:.0f}k.png", dpi=150)
plt.show()
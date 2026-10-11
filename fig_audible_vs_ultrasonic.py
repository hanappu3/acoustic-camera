import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"  # Windowsの日本語フォント

fs = 1_000_000                 # 1 MHzで描画（40 kHzを滑らかに描くため）
t = np.arange(0, 1e-3, 1 / fs)  # 0〜1 ms

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

for ax, f, label, color in [
    (axes[0], 1_000,  "1 kHz（聞こえる音）",       "tab:blue"),
    (axes[1], 40_000, "40 kHz（聞こえない音：エア漏れ）", "tab:red"),
]:
    ax.plot(t * 1e3, np.sin(2 * np.pi * f * t), color=color, linewidth=1)
    ax.set_title(label)
    ax.set_ylabel("振幅")
    ax.set_ylim(-1.3, 1.3)
    ax.grid(alpha=0.3)

axes[1].set_xlabel("時間 [ms]")
fig.suptitle("人の耳は約20 kHzまで。エア漏れの音はその上にある", fontsize=13)
fig.tight_layout()
fig.savefig("fig_audible_vs_ultrasonic.png", dpi=150)
plt.show()
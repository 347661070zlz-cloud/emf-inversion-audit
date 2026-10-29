r"""生成投稿用的出版级插图（PDF 矢量 + PNG 600dpi）。

学术规范要点：
  * 图内不放标题（标题交给 \caption）；
  * 全英文、Times 字体，字号按最终列宽 3.5 in 设定为 8 pt；
  * 四边框、刻度朝内（IEEE 常见样式），不画网格；
  * 矢量 PDF 优先，同时导出 600 dpi PNG 备用；
  * 配色用 Okabe-Ito 色盲友好色板。
"""
import sys
from pathlib import Path

# 允许从任意目录运行：把仓库根目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from src import config as C
from cv_experiment import add_blocks, load_with_coords

OUT = C.FIG_DIR                      # outputs/figures
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 全局样式
COL_W = 3.5          # IEEE 单栏宽（inch）
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,      # 嵌入 TrueType，避免 Type-3 字体告警
    "ps.fonttype": 42,
})

# Okabe-Ito 色盲友好色板
C_BLUE, C_VERM, C_GREEN, C_ORANGE, C_GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#999999"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def tidy(ax):
    for s in ax.spines.values():
        s.set_linewidth(0.6)


# ---------------------------------------------------------------- Fig.1 协议对比
def fig_protocols():
    d = pd.read_csv(C.OUT_DIR / "protocol" / "protocol_aggregate.csv")
    order = ["random", "stpbcv:20:60", "logo"]
    label = ["Random\nhold-out", "ST-PB-CV\n+ embargo", "LOGO-CV"]
    d = d.set_index("scheme").loc[order]
    colors = [C_VERM, C_BLUE, C_ORANGE]

    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    x = np.arange(len(d))
    ax.bar(x, d["MAE_mean"], yerr=d["MAE_ci95"], width=0.55,
           color=colors, edgecolor="black", linewidth=0.5, capsize=2.5,
           error_kw=dict(elinewidth=0.6, capthick=0.6))
    for xi, (_, r) in zip(x, d.iterrows()):
        ax.text(xi, r["MAE_mean"] + r["MAE_ci95"] + 0.02,
                f'{r["MAE_mean"]:.3f}', ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.set_ylabel("Test MAE (dB)")
    ax.set_ylim(0, 1.72)
    ax.yaxis.set_major_locator(MultipleLocator(0.4))
    tidy(ax)
    save(fig, "fig1_protocols")


# ---------------------------------------------------------------- Fig.2 协议×特征
def fig_protocol_feature():
    d = pd.read_csv(C.OUT_DIR / "band_model" / "band_model_summary_random_stpbcv.csv")
    d = d.pivot(index="variant", columns="protocol", values="MAE_mean")
    d = d.loc[["global", "global+earfcn"]]

    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    x = np.arange(2)
    w = 0.34
    series = [("global", "RSRP only (baseline)", C_GREY, ""),
              ("global+earfcn", "+ EARFCN", C_BLUE, "//")]
    for i, (key, lab, col, hatch) in enumerate(series):
        vals = [d.loc[key, "random"], d.loc[key, "stpbcv"]]
        pos = x + (i - 0.5) * w
        bars = ax.bar(pos, vals, width=w, color=col, edgecolor="black",
                      linewidth=0.5, hatch=hatch, label=lab)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0.966, color=C_VERM, linestyle="--", linewidth=0.8)
    ax.text(1.58, 0.94, "reported 0.97 dB", color=C_VERM, fontsize=6.5,
            ha="right", va="top")
    ax.set_xticks(x)
    ax.set_xticklabels(["Random hold-out", "ST-PB-CV + embargo"])
    ax.set_ylabel("Test MAE (dB)")
    ax.set_xlim(-0.55, 1.62)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
              frameon=False, handlelength=1.5, columnspacing=1.4,
              borderaxespad=0.0)
    tidy(ax)
    save(fig, "fig2_protocol_feature")


# ---------------------------------------------------------------- Fig.3 测区与分块
def fig_study_area():
    df = add_blocks(load_with_coords())
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True)
    for ax, col, tag in [(axes[0], "patch", "(a)"), (axes[1], "stblock", "(b)")]:
        codes, _ = pd.factorize(df[col])
        ax.scatter(df["x_m"], df["y_m"], c=codes % 5, cmap="tab10",
                   s=0.35, alpha=0.55, linewidths=0)
        ax.set_xlabel("$x$ (m)")
        ax.set_aspect("equal")
        ax.set_title(f"{tag} {col.replace('stblock', 'patch × time').replace('patch', 'spatial patch')}",
                     fontsize=7)
        tidy(ax)
    axes[0].set_ylabel("$y$ (m)")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig3_study_area")


# ---------------------------------------------------------------- Fig.4 设备偏置
def fig_device():
    d = pd.read_csv(C.OUT_DIR / "protocol" / "device_shift.csv")
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.plot(d["delta"], d["MAE_raw"], "o-", color=C_VERM, markersize=3.5,
            linewidth=1.0, label="Uncalibrated")
    ax.plot(d["delta"], d["MAE_cal"], "s--", color=C_BLUE, markersize=3.2,
            linewidth=1.0, label="Calibrated (100 samples)")
    ax.axhline(d["MAE_raw"].iloc[0], color=C_GREY, linestyle=":", linewidth=0.7)
    ax.set_xlabel("Inter-device RSRP offset $\\delta$ (dB)")
    ax.set_ylabel("Test MAE (dB)")
    ax.set_xlim(-0.3, 5.3)
    ax.set_ylim(0.9, 2.8)
    ax.legend(loc="upper left", frameon=False)
    tidy(ax)
    save(fig, "fig4_device")


if __name__ == "__main__":
    print("generating figures ->", OUT)
    fig_protocols()
    fig_protocol_feature()
    fig_study_area()
    fig_device()
    print("done")

"""复现论文中的图：Figure 4 / 6 / 7 / 8 / 9。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from . import config as C
from . import data_io

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

BAND_COLORS = {100: "#2f6db3", 1850: "#d1495b", 1300: "#2a9d8f", 38400: "#e08e0b"}


def fig4_boxplot(save: bool = True):
    """Figure 4：四个频段原始 RSRP 的箱线图（含 < -110 dBm 的异常值）。"""
    data, labels, colors = [], [], []
    for earfcn in C.BANDS:
        vals = []
        for rnd, ef in C.ROUNDS:
            if ef == earfcn:
                vals.append(data_io.read_rsrp(rnd, earfcn, raw=True)["cell"].to_numpy())
        data.append(np.concatenate(vals))
        labels.append(f"EARFCN {earfcn}\n({C.BANDS[earfcn]})")
        colors.append(BAND_COLORS[earfcn])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, whis=1.5,
                    medianprops=dict(color="black", lw=1.2),
                    flierprops=dict(marker="o", markersize=2.2, alpha=0.45,
                                    markerfacecolor="#666666", markeredgecolor="none"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(C.OUTLIER_DBM, color="#b00020", ls="--", lw=1.1,
               label=f"outlier threshold ({C.OUTLIER_DBM:.0f} dBm)")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RSRP (dBm)")
    ax.set_title("Figure 4  Box plot of RSRP from the measurement system", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    for i, v in enumerate(data):
        n_out = int((v <= C.OUTLIER_DBM).sum())
        ax.annotate(f"n={len(v)} / outlier={n_out}", (i + 1, 0.02),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7, color="#333333")
    fig.tight_layout()
    if save:
        fig.savefig(C.FIG_DIR / "fig4_boxplot.png", bbox_inches="tight")
    return fig


def fig6_scatter(save: bool = True):
    """Figure 6：RSRP 与所选电场强度统计量的散点图（四频段）。"""
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6))
    for ax, earfcn in zip(axes.ravel(), C.BANDS):
        df = data_io.read_rsrp_efield_merged(earfcn)
        stat = C.E_FIELD_FEATURE[earfcn]
        x, y = df["cell"].to_numpy(float), df[stat].to_numpy(float)
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=5, alpha=0.35, c=BAND_COLORS[earfcn], edgecolors="none")
        k, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, k * xs + b, color="#333333", lw=1.0, ls="--")
        ax.set_title(f"EARFCN {earfcn} ({C.BANDS[earfcn]})\n"
                     f"{C.E_FIELD_FEATURE_EN[stat]} vs RSRP, r = {r:.2f}", fontsize=9)
        ax.set_xlabel("RSRP (dBm)")
        ax.set_ylabel("E-field (dBµV/m)")
    fig.suptitle("Figure 6  Scatter of RSRP and electric field strength", fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(C.FIG_DIR / "fig6_scatter.png", bbox_inches="tight")
    return fig


def fig7_cdf(pred: pd.DataFrame, save: bool = True):
    """Figure 7：训练集 / 测试集预测误差的累积分布函数。"""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fracs = {}
    for split, color, label in [("train", "#2f6db3", "Training set"),
                                ("test", "#d1495b", "Test set")]:
        e = np.sort(pred.loc[pred["split"] == split, "abs_err"].to_numpy())
        cdf = np.arange(1, len(e) + 1) / len(e)
        ax.plot(e, cdf, color=color, lw=1.5, label=label)
        fracs[split] = float((e <= 2).mean())
        ax.plot([2], [fracs[split]], "o", ms=4, color=color)
    ax.axvline(2.0, color="#999999", ls=":", lw=1.0)
    ax.annotate(f"|error| ≤ 2 dB\n  training set: {fracs['train']*100:.1f}%\n"
                f"  test set: {fracs['test']*100:.1f}%",
                xy=(2.0, 0.5), xytext=(3.4, 0.45), fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f7f9fb", ec="#ccd6e0"),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Absolute prediction error (dB)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Figure 7  CDF of the error in training and test set", fontsize=10)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    if save:
        fig.savefig(C.FIG_DIR / "fig7_cdf.png", bbox_inches="tight")
    return fig


def fig8_scatter(pred: pd.DataFrame, save: bool = True):
    """Figure 8：测试集真实值 vs 预测值。"""
    d = pred[pred["split"] == "test"]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    hb = ax.hexbin(d["y_true"], d["y_pred"], gridsize=55, cmap="viridis", mincnt=1, bins="log")
    lo = min(d["y_true"].min(), d["y_pred"].min())
    hi = max(d["y_true"].max(), d["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], color="#d1495b", lw=1.2, ls="--", label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Real E-field (dBµV/m)")
    ax.set_ylabel("Predicted E-field (dBµV/m)")
    ax.set_title("Figure 8  Real vs predicted electric field strength (test set)", fontsize=9.5)
    cb = fig.colorbar(hb, ax=ax, shrink=0.85)
    cb.set_label("log10(count)", fontsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(C.FIG_DIR / "fig8_scatter.png", bbox_inches="tight")
    return fig


def fig9_series(pred: pd.DataFrame, n: int = 200, save: bool = True):
    """Figure 9：测试集前 n 个样本的真实值与预测值对比。

    按论文口径取 y_test[:200]（sklearn 返回的测试集数组顺序，本身是打乱的），
    而不是"数据集里最早的那 n 条"——后者会得到一段单频段的连续时间窗，
    动态范围只有 4~5 dB，纵向被放大后看起来像存在系统偏差。
    """
    d = (pred[pred["split"] == "test"].sort_values("test_rank").head(n)
         .reset_index(drop=True))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(d.index, d["y_true"], color="#2f6db3", lw=1.5, label="Real value")
    ax.plot(d.index, d["y_pred"], color="#d1495b", lw=1.2, ls="--", label="Predicted value")
    ax.set_xlabel("Number of the samples")
    ax.set_ylabel("E-field strength (dBµV/m)")
    mae = float(d["abs_err"].mean())
    bias = float((d["y_pred"] - d["y_true"]).mean())
    ax.set_title(f"Figure 9  Real and predicted E-field strength (first {n} test samples, "
                 f"MAE = {mae:.2f} dB, bias = {bias:+.2f} dB)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    ax.set_xlim(0, len(d) - 1)
    fig.tight_layout()
    if save:
        fig.savefig(C.FIG_DIR / "fig9_series.png", bbox_inches="tight")
    return fig


def fig9_convention(pred: pd.DataFrame, n: int = 200, save: bool = True):
    """Figure 9 的取样口径对照：同一模型、同一测试集，只是取的 200 个样本不同。"""
    test = pred[pred["split"] == "test"]
    d_paper = test.sort_values("test_rank").head(n).reset_index(drop=True)
    d_time = test.sort_values("order").head(n).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, d, title in [
        (axes[0], d_time, "(a) 数据集原始顺序：一段连续时间窗\n（全部来自 EARFCN100）"),
        (axes[1], d_paper, "(b) 论文口径：y_test[:200]\n（打乱顺序，跨频段）"),
    ]:
        ax.plot(d.index, d["y_true"], color="#2f6db3", lw=1.5, label="Real value")
        ax.plot(d.index, d["y_pred"], color="#d1495b", lw=1.2, ls="--", label="Predicted value")
        span = d["y_true"].max() - d["y_true"].min()
        bias = (d["y_pred"] - d["y_true"]).mean()
        ax.set_title(f"{title}\n跨度 {span:.1f} dB，偏差 {bias:+.2f} dB", fontsize=8.5)
        ax.set_xlabel("Number of the samples")
    axes[0].set_ylabel("E-field strength (dBµV/m)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Same model, same test set — only the 200 samples differ", fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(C.FIG_DIR / "fig9_convention.png", bbox_inches="tight")
    return fig


def fig9_temporal(pred: pd.DataFrame, n: int = 200, save: bool = True):
    """按时间连续的 200 个样本，四个频段各一格。

    这才是"模型能不能跟住场强随时间的变化"的直接检验：
    打乱顺序的图里，相邻点来自不同频段，曲线的大幅跳动主要来自频段间的电平差。
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.6))
    for ax, earfcn in zip(axes.ravel(), C.BANDS):
        d = (pred[(pred["split"] == "test") & (pred["earfcn"] == earfcn)]
             .sort_values("order").head(n).reset_index(drop=True))
        y, p = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        mae = np.abs(p - y).mean()
        ax.plot(d.index, y, color="#2f6db3", lw=1.4, label="Real value")
        ax.plot(d.index, p, color="#d1495b", lw=1.2, ls="--", label="Predicted value")
        ax.set_title(f"EARFCN {earfcn} ({C.BANDS[earfcn]})   "
                     f"R² = {r2:.2f},  MAE = {mae:.2f} dB,  σ(y) = {y.std():.2f} dB",
                     fontsize=8.5)
        ax.set_xlabel("Sample index (consecutive in time)", fontsize=8)
        ax.set_ylabel("E-field (dBµV/m)", fontsize=8)
        ax.tick_params(labelsize=7.5)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("Figure 9'  Consecutive 200 samples per band — can the model follow the "
                 "field strength over time?", fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(C.FIG_DIR / "fig9_temporal.png", bbox_inches="tight")
    return fig


def fig_feature_importance(model, save: bool = True):
    """附加图：23 个特征的相对重要性。"""
    imp = pd.Series(model.feature_importances_, index=C.FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.barh(imp.index, imp.values, color="#2f6db3", alpha=0.8)
    ax.set_xlabel("Relative importance")
    ax.set_title("GBDT feature importance (23 features)", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    if save:
        fig.savefig(C.FIG_DIR / "fig_feature_importance.png", bbox_inches="tight")
    return fig

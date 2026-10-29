"""Diagnostic figures for the manuscript.

  fig_field_cdf.pdf      CDF of the field strength: random split vs LOGO-CV groups
  fig_pred_scatter.pdf   predicted vs measured field strength, random vs LOGO-CV
  fig_residual_band.pdf  residual distribution per band, with and without EARFCN

Requires outputs/tables/predictions_released.csv, produced by run_all.py.

用法：
    python make_figures_diagnostics.py              # 全部
    python make_figures_diagnostics.py --skip-logo  # 复用已缓存的 LOGO 预测
"""
import argparse
import time

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config as C
from src import model
from cv_experiment import load_with_coords, add_blocks

OUT = C.FIG_DIR                      # outputs/figures
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
})

BAND_COLORS = {100: "#2f6db3", 1850: "#d1495b", 1300: "#2a9d8f", 38400: "#e08e0b"}
GREY = "#4a4a4a"


def ecdf(a):
    a = np.sort(np.asarray(a, dtype=float))
    return a, np.arange(1, len(a) + 1) / len(a)


# --------------------------------------------------------------- LOGO 预测
def logo_predictions(df: pd.DataFrame, jobs: int = 8) -> pd.DataFrame:
    cache = C.TAB_DIR / "predictions_logo.csv"
    if cache.exists():
        d = pd.read_csv(cache)
        if len(d) == len(df):
            print(f"复用缓存 {cache}")
            return d

    logo = LeaveOneGroupOut()
    splits = list(logo.split(df, groups=df["rband"]))

    def one(k, tr_idx, te_idx):
        tr, te = df.loc[tr_idx], df.loc[te_idx]
        m = model.make_model()
        m.set_params(random_state=42)
        m.fit(tr[C.FEATURE_COLS], tr["label"])
        p = m.predict(te[C.FEATURE_COLS])
        return pd.DataFrame({"y_true": te["label"].to_numpy(), "y_pred": p,
                             "earfcn": te["earfcn"].to_numpy(),
                             "fold": k, "rband": te["rband"].to_numpy()})

    t0 = time.time()
    parts = Parallel(n_jobs=jobs)(delayed(one)(k, tr, te)
                                  for k, (tr, te) in enumerate(splits))
    d = pd.concat(parts, ignore_index=True)
    d["abs_err"] = (d["y_true"] - d["y_pred"]).abs()
    d.to_csv(cache, index=False)
    print(f"LOGO {len(splits)} 折，{time.time()-t0:.0f} s -> {cache}")
    return d


def pooled_metrics(d: pd.DataFrame) -> dict:
    y, p = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p),
            "bias": float((p - y).mean())}


# --------------------------------------------------------------- 图 A：CDF
def fig_field_cdf(df: pd.DataFrame, pred_rand: pd.DataFrame):
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))

    # (a) 随机 80/20：训练 vs 测试
    ax = axes[0]
    tr = pred_rand.loc[pred_rand["split"] == "train", "y_true"].to_numpy()
    te = pred_rand.loc[pred_rand["split"] == "test", "y_true"].to_numpy()
    ks = stats.ks_2samp(tr, te)
    for y, c, lab in [(tr, "#2f6db3", f"Training ({len(tr):,})"),
                      (te, "#d1495b", f"Test ({len(te):,})")]:
        xs, ys = ecdf(y)
        ax.plot(xs, ys, color=c, lw=1.5, label=lab)
    # 统计量放进标题，避免压在曲线上
    ax.set_title("(a) Random 80/20 split\n"
                 f"KS $D = {ks.statistic:.3f}$, $p = {ks.pvalue:.2f}$", fontsize=8.5)

    # (b) 留一场次：10 个 (campaign, band) 组 vs 全体
    ax = axes[1]
    for rb, g in df.groupby("rband"):
        ef = int(g["earfcn"].iloc[0])
        xs, ys = ecdf(g["label"])
        ax.plot(xs, ys, color=BAND_COLORS[ef], lw=0.85, alpha=0.85, zorder=3)
    xs, ys = ecdf(df["label"])
    ax.plot(xs, ys, color=GREY, lw=2.2, zorder=5)
    means = df.groupby("rband")["label"].mean()
    sig = df.groupby("rband")["label"].std()
    ax.set_title("(b) LOGO-CV groups, i.e. (campaign, band)\n"
                 f"span {means.max()-means.min():.1f} dB, within-group "
                 f"$\\sigma = {sig.min():.2f}$--${sig.max():.2f}$ dB", fontsize=8.5)

    for ax in axes:
        ax.set_xlabel("Electric field strength (dB$\\mu$V/m)")
        ax.set_ylabel("Cumulative probability")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0, 1.02)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7.5)

    # 图例移到画布下方，彻底避开曲线
    handles = [Line2D([], [], color=GREY, lw=2.2, label="All samples")]
    handles += [Line2D([], [], color=BAND_COLORS[ef], lw=1.4, label=f"EARFCN {ef}")
                for ef in C.BANDS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.3, columnspacing=1.6, handletextpad=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_field_cdf.{ext}", bbox_inches="tight")
    return fig


# --------------------------------------------------------------- 图 B：散点
def fig_scatter(pred_rand: pd.DataFrame, pred_logo: pd.DataFrame):
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35))
    lo, hi = 84.5, 107.5

    panels = [
        (axes[0], pred_rand.loc[pred_rand["split"] == "test"], "(a) Random hold-out"),
        (axes[1], pred_logo, "(b) LOGO-CV"),
    ]
    for ax, d, title in panels:
        ax.hexbin(d["y_true"], d["y_pred"], gridsize=48, cmap="Blues",
                  mincnt=1, bins="log", extent=(lo, hi, lo, hi),
                  linewidths=0.15, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="#b00020", lw=1.0, ls="--", zorder=4)
        for ef, g in d.groupby("earfcn"):
            ax.plot(g["y_true"].mean(), g["y_pred"].mean(), "s", ms=6.5,
                    mfc="white", mec=BAND_COLORS[ef], mew=1.6, zorder=6)
        m = pooled_metrics(d)
        ax.annotate(f"MAE = {m['MAE']:.2f} dB\n$R^2$ = {m['R2']:.3f}\n"
                    f"bias = {m['bias']:+.2f} dB",
                    xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7.8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccd6e0", alpha=0.92))
        ax.set_title(title, fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Measured E-field (dB$\\mu$V/m)")

    handles = [Line2D([], [], marker="s", ls="none", ms=6, mfc="white",
                      mec=BAND_COLORS[ef], mew=1.6, label=f"{ef}")
               for ef in C.BANDS]
    axes[0].legend(handles=handles, loc="lower right", frameon=True, fontsize=6.5,
                   ncol=2, title="band mean (EARFCN)", title_fontsize=6.5,
                   handletextpad=0.4, columnspacing=0.8, labelspacing=0.3,
                   framealpha=0.92, edgecolor="#ccd6e0")
    axes[0].set_ylabel("Predicted E-field (dB$\\mu$V/m)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_pred_scatter.{ext}", bbox_inches="tight")
    return fig


def band_aware_predictions(df: pd.DataFrame, pred_rand: pd.DataFrame) -> pd.DataFrame:
    """在与 predictions_released 相同的随机划分上，训练带 earfcn 特征的模型。"""
    cache = C.TAB_DIR / "predictions_bandaware.csv"
    if cache.exists():
        d = pd.read_csv(cache)
        if len(d) == len(pred_rand):
            print(f"复用缓存 {cache}")
            return d

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)

    Xtr = df.iloc[tr_idx][C.FEATURE_COLS].copy()
    Xte = df.iloc[te_idx][C.FEATURE_COLS].copy()
    Xtr["earfcn"] = df.iloc[tr_idx]["earfcn"].astype(float)
    Xte["earfcn"] = df.iloc[te_idx]["earfcn"].astype(float)

    m = model.make_model()
    m.fit(Xtr, df.iloc[tr_idx]["label"])
    p = m.predict(Xte)

    d = pd.DataFrame({"y_true": df.iloc[te_idx]["label"].to_numpy(), "y_pred": p,
                      "earfcn": df.iloc[te_idx]["earfcn"].to_numpy(),
                      "order": df.iloc[te_idx].index.to_numpy()})
    d["abs_err"] = (d["y_true"] - d["y_pred"]).abs()
    d.to_csv(cache, index=False)

    # 校验划分与 predictions_released 一致
    ref = pred_rand.loc[pred_rand["split"] == "test", "order"].to_numpy()
    same = np.array_equal(np.sort(d["order"].to_numpy()), np.sort(ref))
    print(f"划分与复现表一致: {same} | 测试样本 {len(d):,}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-logo", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    df = load_with_coords()
    df = add_blocks(df)
    df["rband"] = df["round"].astype(str) + "_" + df["earfcn"].astype(str)
    print(f"样本 {len(df):,} | 组 {df['rband'].nunique()}")

    pred_rand = pd.read_csv(C.TAB_DIR / "predictions_released.csv")

    pred_logo = logo_predictions(df, args.jobs) if not args.skip_logo else None
    if pred_logo is None:
        pred_logo = pd.read_csv(C.TAB_DIR / "predictions_logo.csv")

    fig_field_cdf(df, pred_rand)
    fig_scatter(pred_rand, pred_logo)
    pred_bw = band_aware_predictions(df, pred_rand)
    fig_residual_box(pred_rand, pred_bw)

    # 数字汇总，供正文引用
    print("\n=== 关键数字 ===")
    tr = pred_rand.loc[pred_rand["split"] == "train", "y_true"]
    te = pred_rand.loc[pred_rand["split"] == "test", "y_true"]
    ks = stats.ks_2samp(tr, te)
    print(f"随机划分 KS = {ks.statistic:.4f}, p = {ks.pvalue:.3f}")
    means = df.groupby("rband")["label"].mean()
    sig = df.groupby("rband")["label"].std()
    print(f"组均值跨度 {means.max()-means.min():.2f} dB, 组内 σ {sig.min():.2f}--{sig.max():.2f} dB")
    for name, d in [("RH test", pred_rand[pred_rand["split"] == "test"]),
                    ("LOGO", pred_logo)]:
        m = pooled_metrics(d)
        print(f"{name}: MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} "
              f"R2={m['R2']:.3f} bias={m['bias']:+.3f}")
    # 逐频段偏差（LOGO 下是否系统性偏移）
    print("\nLOGO 逐频段 bias:")
    for ef, g in pred_logo.groupby("earfcn"):
        print(f"  EARFCN {ef:5d}: bias {(g.y_pred-g.y_true).mean():+.2f} dB, "
              f"MAE {(g.y_pred-g.y_true).abs().mean():.2f} dB")
    print("\nRH 逐频段 bias:")
    d = pred_rand[pred_rand["split"] == "test"]
    for ef, g in d.groupby("earfcn"):
        print(f"  EARFCN {ef:5d}: bias {(g.y_pred-g.y_true).mean():+.2f} dB, "
              f"MAE {(g.y_pred-g.y_true).abs().mean():.2f} dB")

    print(f"\n用时 {time.time()-t0:.0f} s -> {OUT}")




# ------------------------------------------- 图 E：逐频段残差分布（全测试集）
def fig_residual_box(pred_rand: pd.DataFrame, pred_bw: pd.DataFrame):
    from matplotlib.lines import Line2D

    te = pred_rand[pred_rand["split"] == "test"].set_index("order")
    bw = pred_bw.set_index("order")

    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    off = 0.17
    bias = []
    for i, ef in enumerate(C.BANDS):
        o = te[te["earfcn"] == ef].index
        y = te.loc[o, "y_true"].to_numpy()
        r0 = te.loc[o, "y_pred"].to_numpy() - y
        r1 = bw.loc[o, "y_pred"].to_numpy() - y
        bias.append((ef, r0.mean(), r1.mean()))
        for dx, r, c in [(-off, r0, "#8a8a8a"), (+off, r1, "#d1495b")]:
            bp = ax.boxplot([r], positions=[i + 1 + dx], widths=0.3,
                            patch_artist=True, showfliers=False, whis=1.5,
                            medianprops=dict(color="black", lw=1.0),
                            boxprops=dict(facecolor=c, alpha=0.55, lw=0.8),
                            whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
            ax.plot(i + 1 + dx, r.mean(), "D", ms=4, mfc="white",
                    mec="black", mew=1.0, zorder=5)

    ax.axhline(0, color="#b00020", lw=0.9, ls="--", zorder=1)
    ax.set_xticks(range(1, len(C.BANDS) + 1))
    ax.set_xticklabels([f"EARFCN {ef}\nbias {b0:+.2f} $\\rightarrow$ {b1:+.2f}"
                        for ef, b0, b1 in bias], fontsize=7.5)
    ax.set_ylabel("Residual (predicted $-$ measured) (dB)")
    ax.set_ylim(-4.1, 4.35)

    handles = [Line2D([], [], marker="s", ls="none", ms=8, mfc="#8a8a8a",
                      mec="#8a8a8a", alpha=0.7, label="Baseline (RSRP only)"),
               Line2D([], [], marker="s", ls="none", ms=8, mfc="#d1495b",
                      mec="#d1495b", alpha=0.7, label="$+$ EARFCN"),
               Line2D([], [], marker="D", ls="none", ms=4, mfc="white",
                      mec="black", label="mean (bias)")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.2,
              ncol=3, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_residual_band.{ext}", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main()

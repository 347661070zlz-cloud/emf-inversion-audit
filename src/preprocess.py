"""预处理：异常值剔除、EWMA 滤波、分段线性插值，以及 Table 1/2/3 的统计。

对应论文 3.2 节：
  3.2.1 用箱线图思想剔除 RSRP < -110 dBm 的异常值
  3.2.2 用 Pearson 相关系数从 7 个电场强度统计量中挑一个作为标签
  3.2.3 EWMA 滤波（式 2）
  3.2.4 分段线性插值，仅填补 N = 5 s 以内的缺失
"""
import numpy as np
import pandas as pd

from . import config as C
from . import data_io


# ------------------------------------------------------------------ 基础算子
def remove_outliers(rsrp: pd.DataFrame, threshold: float = C.OUTLIER_DBM) -> pd.DataFrame:
    """剔除低于阈值的 RSRP（论文：低于 -110 dBm 视为噪声主导的无信号状态）。"""
    return rsrp[rsrp["cell"] > threshold].copy()


def ewma(series: pd.Series, alpha: float = C.EWMA_ALPHA) -> pd.Series:
    """指数加权移动平均：EWMA_t = a*X_t + (1-a)*EWMA_{t-1}（式 2）。"""
    return series.ewm(alpha=alpha, adjust=False).mean()


def to_grid(series: pd.Series, index: pd.DatetimeIndex, limit: int = C.INTERP_LIMIT) -> pd.Series:
    """重采样到 1 s 网格并做分段线性插值，仅填补 limit 秒以内的空洞。"""
    g = series.reindex(index)
    return g.interpolate(method="linear", limit=limit, limit_area="inside")


def common_grid(rsrp: pd.DataFrame, efield: pd.DataFrame, freq: str = C.GRID_FREQ) -> pd.DatetimeIndex:
    """两条序列时间重叠区间的 1 s 网格。"""
    t0 = max(rsrp["t"].min(), efield["t"].min())
    t1 = min(rsrp["t"].max(), efield["t"].max())
    return pd.date_range(t0, t1, freq=freq)


def match_pairs(rsrp: pd.DataFrame, efield: pd.DataFrame,
                tolerance: str = "3s") -> pd.DataFrame:
    """以电场强度的时间戳为基准，为每条记录匹配最近的 RSRP（论文图 5 的对齐方式）。"""
    left = efield[["t"] + C.E_FIELD_STATS].drop_duplicates("t").copy()
    right = rsrp[["t", "cell"]].drop_duplicates("t").copy()
    m = pd.merge_asof(left.sort_values("t"), right.sort_values("t"),
                      on="t", direction="nearest", tolerance=pd.Timedelta(tolerance))
    return m.dropna(subset=["cell"]).reset_index(drop=True)


# ------------------------------------------------------------------ Table 1
def table1_pearson() -> pd.DataFrame:
    """Table 1：RSRP 与 7 个电场强度统计量的 Pearson 相关系数（原始数据，未滤波）。"""
    rows = []
    for rnd, earfcn in C.ROUNDS:
        rsrp = data_io.read_rsrp(rnd, earfcn)
        ef = data_io.read_efield(rnd, earfcn)
        m = match_pairs(rsrp, ef, tolerance="1s")
        row = {"Round": rnd, "EARFCN": earfcn, "n_pairs": len(m)}
        for s in C.E_FIELD_STATS:
            row[s] = round(float(m["cell"].corr(m[s])), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def selected_stat_table1() -> pd.DataFrame:
    """按 Table 1 结果，为每个频段选出相关性最高的统计量并给出对应 r。"""
    t1 = table1_pearson()
    out = []
    for earfcn, cols in t1.groupby("EARFCN"):
        best = cols[C.E_FIELD_STATS].mean().idxmax()
        out.append({"EARFCN": earfcn, "selected": best,
                    "r_mean": round(cols[best].mean(), 3)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ Table 3
def table3_pearson(alpha: float = C.EWMA_ALPHA) -> pd.DataFrame:
    """Table 3：预处理各阶段 RSRP 与所选电场强度统计量的相关系数。

    Raw           : 原始数据按时间戳配对后的相关系数
    EWMA          : 配对后两条序列分别做 EWMA（论文式 2）后的相关系数
    Interpolation : 再重采样到 1 s 网格并分段线性插值（N=5 s）后的相关系数
    """
    rows = []
    for rnd, earfcn in C.ROUNDS:
        rsrp = remove_outliers(data_io.read_rsrp(rnd, earfcn))
        ef = data_io.read_efield(rnd, earfcn)
        stat = C.E_FIELD_FEATURE[earfcn]

        m = match_pairs(rsrp, ef, tolerance="5s")
        r_raw = float(m["cell"].corr(m[stat]))

        m = m.assign(rsrp_f=ewma(m["cell"], alpha), efield_f=ewma(m[stat], alpha))
        r_ewma = float(m["rsrp_f"].corr(m["efield_f"]))

        idx = pd.date_range(m["t"].min(), m["t"].max(), freq=C.GRID_FREQ)
        g = m.set_index("t")[["rsrp_f", "efield_f"]].reindex(idx).interpolate(
            method="linear", limit=C.INTERP_LIMIT, limit_area="inside")
        r_int = float(g["rsrp_f"].corr(g["efield_f"]))

        rows.append({"Round": rnd, "EARFCN": earfcn, "stat": stat,
                     "Raw": round(r_raw, 3), "EWMA": round(r_ewma, 3),
                     "Interpolation": round(r_int, 3), "n_pairs": len(m)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Table 2
def table2_dataset_summary(regen_samples: pd.DataFrame | None = None) -> pd.DataFrame:
    """Table 2：各频段的原始 RSRP 条数、电场强度条数、样本数。"""
    rows = []
    tot = {"rsrp": 0, "efield": 0, "samples": 0}
    for earfcn in C.BANDS:
        n_rsrp = n_ef = 0
        for rnd, ef in C.ROUNDS:
            if ef != earfcn:
                continue
            n_rsrp += len(data_io.read_rsrp(rnd, earfcn, raw=True))
            n_ef += len(data_io.read_efield(rnd, earfcn))
        n_samp = 0 if regen_samples is None else int((regen_samples["earfcn"] == earfcn).sum())
        rows.append({"EARFCN": earfcn, "RSRP (APP)": n_rsrp,
                     "E-field (spectrum)": n_ef, "Samples": n_samp})
        tot["rsrp"] += n_rsrp
        tot["efield"] += n_ef
        tot["samples"] += n_samp
    rows.append({"EARFCN": "Total", "RSRP (APP)": tot["rsrp"],
                 "E-field (spectrum)": tot["efield"], "Samples": tot["samples"]})
    return pd.DataFrame(rows)

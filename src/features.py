"""特征工程：论文 Algorithm 1 + 附录 A。

对每一个电场强度采样时刻 t：
  * 取 1 s 网格上 t-3 ... t+3 共 7 个（滤波后）RSRP 值，计算 16 个统计量；
  * 再把这 7 个（未滤波的）原始 RSRP 值 t1..t7 一并作为特征；
  * 电场强度（所选统计量，滤波后）作为标签 label。
"""
import numpy as np
import pandas as pd

from . import config as C
from . import data_io
from . import preprocess as P


# ------------------------------------------------------------------ 附录 A 的统计量
def _order_stat(x: np.ndarray, p: float) -> float:
    """位置式分位数：第 round(p*(N+1)) 个次序统计量（论文附录 A 的定义）。"""
    n = len(x)
    k = int(round(p * (n + 1))) - 1
    k = min(max(k, 0), n - 1)
    return float(np.sort(x)[k])


def grouped_mode(x: np.ndarray, bin_width: float = 0.5) -> float:
    """分组众数：XMode = L + fb/(fa+fb) * i（附录 A 第 10 项）。"""
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return lo
    edges = np.arange(lo, hi + bin_width, bin_width)
    if len(edges) < 2:
        return lo
    counts, edges = np.histogram(x, bins=edges)
    k = int(np.argmax(counts))
    f_m = counts[k]
    f_b = counts[k - 1] if k > 0 else 0
    f_a = counts[k + 1] if k + 1 < len(counts) else 0
    if f_a + f_b == 0:
        return float(0.5 * (edges[k] + edges[k + 1]))
    return float(edges[k] + f_b / (f_b + f_a) * (edges[k + 1] - edges[k]))


def window_stats(w: np.ndarray) -> dict:
    """一个 7 点窗口的 16 个统计特征（附录 A Table A1）。"""
    n = len(w)
    mean = float(np.mean(w))
    std = float(np.std(w, ddof=0))
    mx, mn = float(np.max(w)), float(np.min(w))
    med = _order_stat(w, 0.5)
    q1 = _order_stat(w, 0.25)
    q3 = _order_stat(w, 0.75)
    z = (w - mean) / std if std > 1e-12 else np.zeros_like(w)
    return {
        "mean_cell": mean,
        "max_cell": mx,
        "min_cell": mn,
        "median_cell": med,
        "quantile1_cell": q1,
        "quantile2_cell": q3,          # 论文中 quantile2 即第三四分位数
        "range_cell": mx - mn,
        "std": std,
        "variation_cell": float(np.var(w, ddof=0)),   # 方差
        "mode_cell": grouped_mode(w),
        "skew_cell": float(np.sum(z ** 3) / (n - 1)),
        "kurt_cell": float(np.sum(z ** 4) / (n - 1)),
        "mean_mad": float(np.mean(np.abs(w - mean))),
        "median_mad": float(np.mean(np.abs(w - med))),
        "iqr_cell": q3 - q1,
        "cv": std / mean if abs(mean) > 1e-12 else np.nan,
    }


# ------------------------------------------------------------------ 切换检测
def detect_switch_events(rsrp: pd.DataFrame, jump_db: float = 10.0,
                         gap_s: float = 5.0, merge_s: float = 10.0) -> list:
    """识别"服务小区/频段切换"时刻。

    数据按频段拆分后，同一文件内仍有两类事件：
      1) 时间中断（dt > gap_s）——手机短暂驻留到别的频段/小区，本频段没有记录；
      2) 电平突跳（相邻两点 |ΔRSRP| > jump_db）——发生了切换（handover）。
    两者都会让 RSRP 序列在切换点前后不连续；7 点窗口一旦跨过切换点，
    特征就混入了两段不同信道的样本，与标签的定义不再一致。
    """
    d = rsrp.sort_values("t")
    dt = d["t"].diff().dt.total_seconds()
    jump = d["cell"].diff().abs()
    times = d.loc[(dt > gap_s) | (jump > jump_db), "t"].tolist()

    merged = []
    for t in times:
        if merged and (t - merged[-1]).total_seconds() <= merge_s:
            continue
        merged.append(t)
    return merged


def guard_mask(index: pd.DatetimeIndex, events: list,
               guard_s: float, half: int) -> np.ndarray:
    """保留掩码：距任一事件 <= guard_s + half 的样本剔除。

    +half 是因为样本特征取自 t-half ... t+half 的 7 点窗口，
    窗口一旦跨过切换点，该样本就不可信。
    """
    keep = np.ones(len(index), dtype=bool)
    span = pd.Timedelta(seconds=guard_s + half)
    for te in events:
        keep &= ~((index >= te - span) & (index <= te + span))
    return keep


# ------------------------------------------------------------------ Algorithm 1
def build_samples(round_id: int, earfcn: int, alpha: float = C.EWMA_ALPHA,
                  limit: int = C.INTERP_LIMIT, half: int = C.WINDOW_HALF,
                  guard_s: float = 0.0, jump_db: float = 10.0) -> pd.DataFrame:
    """对单轮单频段执行 Algorithm 1，返回 23 特征 + label 的样本表。

    guard_s > 0 时剔除切换事件前后 guard_s 秒内的样本（保护间隔）。
    """
    rsrp = P.remove_outliers(data_io.read_rsrp(round_id, earfcn))
    efield = data_io.read_efield(round_id, earfcn)
    stat = C.E_FIELD_FEATURE[earfcn]

    events = detect_switch_events(rsrp, jump_db=jump_db) if guard_s > 0 else []
    idx = P.common_grid(rsrp, efield)
    rsrp_sparse = rsrp.drop_duplicates("t").set_index("t")["cell"].astype(float).reindex(idx)
    efield_sparse = efield.drop_duplicates("t").set_index("t")[stat].astype(float).reindex(idx)

    # 滤波 + 插值（论文 3.2.3、3.2.4）：先对原始采样序列做 EWMA，再线性插值到 1 s 网格
    rsrp_f = P.to_grid(P.ewma(rsrp_sparse, alpha), idx, limit)
    efield_f = P.to_grid(P.ewma(efield_sparse, alpha), idx, limit)
    rsrp_raw = P.to_grid(rsrp_sparse, idx, limit)

    # 7 点滑动窗口，列顺序 t-3 ... t+3
    cols = [rsrp_f.shift(k) for k in range(half, -half - 1, -1)]
    W = pd.concat(cols, axis=1)
    W.columns = [f"t{i + 1}" for i in range(len(cols))]
    raw_cols = [rsrp_raw.shift(k) for k in range(half, -half - 1, -1)]
    Wr = pd.concat(raw_cols, axis=1)
    Wr.columns = [f"t{i + 1}" for i in range(len(raw_cols))]

    recs = []
    for ts, row in W.iterrows():
        w = row.to_numpy(dtype=float)
        if np.isnan(w).any():
            continue
        d = window_stats(w)
        raw = Wr.loc[ts].to_numpy(dtype=float)
        for i in range(len(raw)):
            d[f"t{i + 1}"] = float(raw[i])
        d["label"] = float(efield_f.loc[ts]) if not np.isnan(efield_f.loc[ts]) else np.nan
        d["time"] = ts
        recs.append(d)

    df = pd.DataFrame(recs)
    df = df.dropna(subset=["label"] + C.FEATURE_COLS).reset_index(drop=True)

    if guard_s > 0 and events:
        t = pd.to_datetime(df["time"], format="%Y%m%d-%H%M%S")
        df = df.loc[guard_mask(pd.DatetimeIndex(t), events, guard_s, half)]
        df = df.reset_index(drop=True)

    df.insert(0, "time", df.pop("time").dt.strftime("%Y%m%d-%H%M%S"))
    df.insert(0, "earfcn", earfcn)
    df.insert(0, "round", round_id)
    return df


def build_all(alpha: float = C.EWMA_ALPHA, limit: int = C.INTERP_LIMIT,
              verbose: bool = True, guard_s: float = 0.0,
              jump_db: float = 10.0) -> pd.DataFrame:
    """合并 5 轮 10 组测量，得到完整样本集。guard_s > 0 时启用切换保护间隔。"""
    frames = []
    for rnd, earfcn in C.ROUNDS:
        df = build_samples(rnd, earfcn, alpha=alpha, limit=limit,
                           guard_s=guard_s, jump_db=jump_db)
        frames.append(df)
        if verbose:
            print(f"  round{rnd} earfcn{earfcn:<5d} -> {len(df):6d} samples")
    return pd.concat(frames, ignore_index=True)


def compare_with_released(regen: pd.DataFrame, released: pd.DataFrame) -> pd.DataFrame:
    """逐列比较重建特征与作者发布样本的一致性（按 round+earfcn+time 对齐）。"""
    key = ["round", "earfcn", "time"]
    m = regen.merge(released[key + C.FEATURE_COLS + ["label"]], on=key, suffixes=("_regen", ""))
    rows = []
    for col in C.FEATURE_COLS + ["label"]:
        a = m[f"{col}_regen"] if f"{col}_regen" in m else m[col]
        b = m[col]
        group = "label" if col == "label" else ("raw RSRP (t1-t7)" if col.startswith("t") else "statistic (16)")
        rows.append({"group": group, "feature": col, "n": len(m),
                     "MAE": float(np.nanmean(np.abs(a - b))),
                     "corr": float(np.corrcoef(a.fillna(a.mean()), b.fillna(b.mean()))[0, 1])})
    return pd.DataFrame(rows)

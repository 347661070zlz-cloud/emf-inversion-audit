"""不同数据划分方式对反演性能与可信度的影响。

背景
----
论文用的是**随机 80/20 划分**。但这批数据是"边走边测"的 1 s 时序 + 空间轨迹：
相邻样本在时间和空间上都高度相关，随机划分会让测试点的"邻居"落进训练集，
等于把答案提前告诉模型，指标会明显偏乐观。

本脚本对比四种划分：

1. random   ：随机 80/20（论文做法，8 个随机种子）
2. pbcv     ：Patch-Based CV —— 把测区按 50 m 网格切块，整块进同一折；
              训练集额外剔除距任一测试点 20 m 内的样本（缓冲区，防止边界泄漏）
3. stpbcv   ：Spatio-Temporal Patch-Based CV —— 块 = (50 m 网格 × 10 min 时间窗)；
              缓冲区 = 空间 20 m 且时间 60 s 内
4. route    ：按测量轮次留一法（5 折）——检验"换一次测量"能否泛化

用法：
    python cv_experiment.py                     # 全部四种
    python cv_experiment.py --schemes random pbcv
"""
import argparse
import json
import time

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

from src import config as C
from src import data_io, model

OUT = C.OUT_DIR / "cv"
OUT.mkdir(parents=True, exist_ok=True)
COORD_CACHE = OUT / "samples_with_coords.csv"

PATCH_M = 50.0      # 空间块边长（米）
BLOCK_S = 600.0     # 时间块长度（秒）
BUF_M = 20.0        # 空间缓冲区（米）
BUF_S = 60.0        # 时间缓冲区（秒）


# ------------------------------------------------------------------ 数据准备
def load_with_coords() -> pd.DataFrame:
    """给每个样本补上经纬度（按时间戳最近邻匹配 RSRP 记录）。"""
    if COORD_CACHE.exists():
        df = pd.read_csv(COORD_CACHE)
        df["t"] = pd.to_datetime(df["time"], format="%Y%m%d-%H%M%S")
        return df

    rel = data_io.read_released_samples()
    parts = []
    for rnd, ef in C.ROUNDS:
        sub = rel[(rel["round"] == rnd) & (rel["earfcn"] == ef)].sort_values("t")
        rp = data_io.read_rsrp(rnd, ef, raw=True).drop_duplicates("t").sort_values("t")
        m = pd.merge_asof(sub, rp[["t", "lat", "lon"]], on="t", direction="nearest",
                          tolerance=pd.Timedelta("3s"))
        parts.append(m)
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    df.to_csv(COORD_CACHE, index=False)
    return df


def add_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """给每个样本打上空间块（50 m 网格）与时空块（×10 min 时间窗）标签。"""
    lat0, lon0 = df["lat"].min(), df["lon"].min()
    dlat = PATCH_M / 111_000.0
    dlon = PATCH_M / (111_000.0 * np.cos(np.radians(df["lat"].mean())))

    df = df.copy()
    df["ix"] = ((df["lat"] - lat0) / dlat).astype(int)
    df["iy"] = ((df["lon"] - lon0) / dlon).astype(int)
    df["patch"] = df["ix"].astype(str) + "_" + df["iy"].astype(str)

    t0 = df["t"].min()
    df["tblock"] = ((df["t"] - t0).dt.total_seconds() // BLOCK_S).astype(int)
    df["stblock"] = df["patch"] + "@" + df["tblock"].astype(str)

    df["x_m"] = (df["lat"] - lat0) * 111_000.0
    df["y_m"] = (df["lon"] - lon0) * 111_000.0 * np.cos(np.radians(df["lat"].mean()))
    return df


def spatial_buffer_mask(tr: pd.DataFrame, te: pd.DataFrame, radius: float = BUF_M,
                        time_gap: float | None = None) -> np.ndarray:
    """返回训练集保留掩码：剔除与测试点空间距离 < radius（且时间接近）的样本。"""
    tree = cKDTree(te[["x_m", "y_m"]].to_numpy())
    cand = tree.query_ball_point(tr[["x_m", "y_m"]].to_numpy(), r=radius)
    keep = np.ones(len(tr), dtype=bool)
    if time_gap is None:
        keep[[i for i, c in enumerate(cand) if len(c) > 0]] = False
        return keep

    tr_t = tr["t"].to_numpy()
    te_t = te["t"].to_numpy()
    for i, idxs in enumerate(cand):
        if not idxs:
            continue
        dt = np.abs((te_t[idxs] - tr_t[i]) / np.timedelta64(1, "s"))
        if (dt < time_gap).any():
            keep[i] = False
    return keep


# ------------------------------------------------------------------ 评估
def fit_eval(Xtr, ytr, Xte, yte, seed=42) -> dict:
    m = model.make_model()
    m.set_params(random_state=seed)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    return {"R2": r2_score(yte, p), "RMSE": float(np.sqrt(mean_squared_error(yte, p))),
            "MAE": mean_absolute_error(yte, p)}


def run_fold(df: pd.DataFrame, tr_idx, te_idx, scheme: str, fold: int,
             use_buffer: bool, time_gap: float | None) -> dict:
    tr = df.loc[tr_idx]
    te = df.loc[te_idx]
    if use_buffer:
        keep = spatial_buffer_mask(tr, te, BUF_M, time_gap)
        tr = tr[keep]
    res = fit_eval(tr[C.FEATURE_COLS], tr["label"], te[C.FEATURE_COLS], te["label"])
    return {"scheme": scheme, "fold": fold, "n_train": len(tr), "n_test": len(te), **res}


def build_splits(df: pd.DataFrame, scheme: str, seeds: int):
    """产出 (fold_id, train_idx, test_idx, time_gap) 列表。"""
    out = []
    if scheme == "random":
        for s in range(seeds):
            tr, te = train_test_split(np.arange(len(df)), test_size=C.TEST_SIZE,
                                      random_state=s)
            out.append((s, tr, te, False, None))
        return out

    groups = {"pbcv": "patch", "stpbcv": "stblock", "route": "round"}[scheme]
    gkf = GroupKFold(n_splits=5)
    for k, (tr, te) in enumerate(gkf.split(df, groups=df[groups])):
        if scheme == "pbcv":
            out.append((k, tr, te, True, None))      # 只用空间缓冲
        elif scheme == "stpbcv":
            out.append((k, tr, te, True, BUF_S))     # 空间 + 时间缓冲
        else:
            out.append((k, tr, te, False, None))     # route：不设缓冲
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+",
                    default=["random", "pbcv", "stpbcv", "route"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    t0 = time.time()
    df = load_with_coords()
    df = add_blocks(df)
    print(f"样本 {len(df):,} 条 | 空间块 {df['patch'].nunique()} 个 | "
          f"时空块 {df['stblock'].nunique()} 个 | 轮次 {df['round'].nunique()} 个")

    jobs = []
    for scheme in args.schemes:
        for k, tr, te, use_buf, gap in build_splits(df, scheme, args.seeds):
            jobs.append((scheme, k, tr, te, use_buf, gap))
    print(f"共 {len(jobs)} 次训练，开始…", flush=True)

    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_fold)(df, tr, te, s, k, buf, gap)
        for s, k, tr, te, buf, gap in jobs)

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(OUT / "cv_per_fold.csv", index=False)

    agg = (per_fold.groupby("scheme")
           .agg(n_folds=("fold", "count"),
                n_train=("n_train", "mean"), n_test=("n_test", "mean"),
                R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"))
           .reset_index())
    order = [s for s in ["random", "route", "pbcv", "stpbcv"] if s in set(agg["scheme"])]
    agg["order"] = agg["scheme"].map({s: i for i, s in enumerate(order)})
    agg = agg.sort_values("order").drop(columns="order")
    agg.to_csv(OUT / "cv_aggregate.csv", index=False)

    print("\n" + agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    base = agg.loc[agg["scheme"] == "random", "MAE_mean"]
    if len(base):
        agg["MAE_gap_vs_random"] = agg["MAE_mean"] - float(base.iloc[0])
        print("\n相对随机划分的 MAE 增幅：")
        for _, r in agg.iterrows():
            print(f"  {r['scheme']:7s} {r['MAE_mean']:.3f} dB "
                  f"({r['MAE_gap_vs_random']:+.3f} dB, "
                  f"R² {r['R2_mean']:.3f})")
    print(f"\n用时 {time.time()-t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()

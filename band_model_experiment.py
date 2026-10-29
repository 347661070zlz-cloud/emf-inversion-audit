"""分频段建模 vs 全局建模：earfcn 作为已知信息该怎么用。

三种方案
--------
1. **global**      ：单一模型，23 维 RSRP 特征（论文做法，模型不知道频段）
2. **global+earfcn**：单一模型，额外把 earfcn 作为数值特征（树可以直接在 4 个频段上切分）
3. **per-band**    ：按 earfcn 训练 4 个独立模型，推理时按当前 earfcn 选模型

评估协议：随机留出（与论文可比）+ ST-PB-CV + 禁运（推荐口径）。

用法：
    python band_model_experiment.py --seeds 3 --jobs 6
"""
import argparse
import json
import time

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split

from src import config as C
from src import model
from cv_experiment import BUF_M, BUF_S, add_blocks, load_with_coords, spatial_buffer_mask

OUT = C.OUT_DIR / "band_model"
OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = ["global", "global+earfcn", "per-band"]


def make_X(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    X = df[C.FEATURE_COLS].copy()
    if variant == "global+earfcn":
        X["earfcn"] = df["earfcn"].astype(float)
    return X


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, variant: str):
    """返回测试集预测值（per-band 时按频段分别建模）。"""
    if variant != "per-band":
        m = model.make_model()
        m.fit(make_X(tr, variant), tr["label"])
        return m.predict(make_X(te, variant))

    pred = np.full(len(te), np.nan)
    te_band = te["earfcn"].to_numpy()
    for ef in np.unique(te_band):
        tr_b = tr[tr["earfcn"] == ef]
        if len(tr_b) < 200:                      # 该频段训练样本太少则退回全局
            m = model.make_model()
            m.fit(make_X(tr, "global"), tr["label"])
            pred[te_band == ef] = m.predict(make_X(te[te_band == ef], "global"))
            continue
        m = model.make_model()
        m.fit(make_X(tr_b, "global"), tr_b["label"])
        pred[te_band == ef] = m.predict(make_X(te[te_band == ef], "global"))
    return pred


def run_one(df, tr_idx, te_idx, variant, protocol, fold):
    tr, te = df.loc[tr_idx], df.loc[te_idx]
    if protocol == "stpbcv":
        tr = tr[spatial_buffer_mask(tr, te, BUF_M, BUF_S)]

    p = fit_predict(tr, te, variant)
    y = te["label"].to_numpy()
    out = {"variant": variant, "protocol": protocol, "fold": fold,
           "n_train": len(tr), "n_test": len(te),
           "MAE": mean_absolute_error(y, p),
           "RMSE": float(np.sqrt(mean_squared_error(y, p))),
           "R2": r2_score(y, p),
           "te_mean": float(y.mean()), "te_std": float(y.std())}
    band = te["earfcn"].to_numpy()
    for ef in np.unique(band):
        mk = band == ef
        out[f"b{ef}_MAE"] = float(np.abs(p[mk] - y[mk]).mean())
        out[f"b{ef}_R2"] = float(1 - ((y[mk] - p[mk]) ** 2).sum()
                                 / ((y[mk] - y[mk].mean()) ** 2).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--protocols", nargs="+", default=["random", "stpbcv"])
    args = ap.parse_args()

    t0 = time.time()
    df = add_blocks(load_with_coords())
    print(f"样本 {len(df):,} | 各频段样本数 {df['earfcn'].value_counts().to_dict()}")

    df["rband"] = df["round"].astype(str) + "_" + df["earfcn"].astype(str)
    jobs = []
    for variant in VARIANTS:
        if "random" in args.protocols:
            for s in range(args.seeds):
                tr, te = train_test_split(np.arange(len(df)), test_size=C.TEST_SIZE,
                                          random_state=s)
                jobs.append((tr, te, variant, "random", s))
        if "stpbcv" in args.protocols:
            gkf = GroupKFold(n_splits=5)
            for k, (tr, te) in enumerate(gkf.split(df, groups=df["stblock"])):
                jobs.append((tr, te, variant, "stpbcv", k))
        if "logo" in args.protocols:
            for k, (tr, te) in enumerate(LeaveOneGroupOut().split(df, groups=df["rband"])):
                jobs.append((tr, te, variant, "logo", k))
    print(f"共 {len(jobs)} 次训练…", flush=True)

    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_one)(df, tr, te, v, p, k) for tr, te, v, p, k in jobs)
    per = pd.DataFrame(rows)
    tag = "_".join(args.protocols)
    per.to_csv(OUT / f"band_model_per_fold_{tag}.csv", index=False)

    # 汇总 + 合并 R²
    agg_rows = []
    for (proto, var), d in per.groupby(["protocol", "variant"]):
        n = d["n_test"].sum()
        sse = (d["n_test"] * d["RMSE"] ** 2).sum()
        grand = (d["n_test"] * d["te_mean"]).sum() / n
        sst = (d["n_test"] * (d["te_std"] ** 2 + (d["te_mean"] - grand) ** 2)).sum()
        row = {"protocol": proto, "variant": var, "n_folds": len(d),
               "MAE_mean": d["MAE"].mean(), "MAE_std": d["MAE"].std(),
               "RMSE_mean": d["RMSE"].mean(), "R2_pooled": 1 - sse / sst}
        for ef in [100, 1850, 1300, 38400]:
            row[f"b{ef}_MAE"] = d[f"b{ef}_MAE"].mean()
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    order = {v: i for i, v in enumerate(VARIANTS)}
    agg["o"] = agg["variant"].map(order)
    agg = agg.sort_values(["protocol", "o"]).drop(columns="o")
    agg.to_csv(OUT / f"band_model_summary_{tag}.csv", index=False)

    print("\n" + agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n相对全局模型的 MAE 变化：")
    for proto, d in agg.groupby("protocol"):
        base = d[d["variant"] == "global"]["MAE_mean"].iloc[0]
        for _, r in d.iterrows():
            print(f"  [{proto:7s}] {r['variant']:14s} {r['MAE_mean']:.4f} dB "
                  f"({r['MAE_mean']-base:+.4f})  pooled R²={r['R2_pooled']:.4f}")
    print(f"\n用时 {time.time()-t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()

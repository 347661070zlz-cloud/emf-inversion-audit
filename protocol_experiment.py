"""三协议对照：Random Hold-out / LOGO-CV / ST-PB-CV（含禁运）。

协议定义
--------
1. **Random Hold-out**：随机 80/20，重复多个随机种子（论文做法）。
   测试点的邻居大概率还在训练集里 → 时空双重泄漏。

2. **LOGO-CV**（Leave-One-Group-Out）：以「测量轮次 × 频段」为组（本数据 5 轮 × 2 频段 = 10 组），
   每次留出一整组做测试，共 10 折。
   - 组内样本来自同一次测量、同一频段，时间上连续；
   - 留出整组后，测试集与训练集**时间不重叠**，但**空间仍重叠**（5 轮走同一片区域）；
   - 分组用 (轮次, 频段) 而不是只用轮次：每轮只测 2 个频段，只按轮次留一会把整个频段
     从训练集删掉，混淆"跨场次"与"跨频段"两个因素。

3. **ST-PB-CV + 禁运**（Spatio-Temporal Patch-Based CV with embargo）：
   块 = (50 m 空间网格 × 10 min 时间窗)，按块分组做 5 折；
   并在训练集上执行**禁运**：剔除与任一测试样本「空间 < 20 m 且时间 < 60 s」的样本，
   切断块边界处的时间自相关与空间邻近泄漏。

用法
----
    python protocol_experiment.py                      # 三协议 + 禁运敏感性
    python protocol_experiment.py --seeds 8 --jobs 6
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
from scipy import stats

from src import config as C
from src import model
from cv_experiment import (BUF_M, BUF_S, add_blocks, load_with_coords,
                           spatial_buffer_mask)

OUT = C.OUT_DIR / "protocol"
OUT.mkdir(parents=True, exist_ok=True)


def parse_spec(spec: str):
    """'stpbcv' / 'stpbcv:20:60' -> (name, buf_m, buf_s)"""
    parts = spec.split(":")
    name = parts[0]
    buf_m = float(parts[1]) if len(parts) > 1 else BUF_M
    buf_s = float(parts[2]) if len(parts) > 2 else BUF_S
    return name, buf_m, buf_s


def fit_eval(Xtr, ytr, Xte, yte, seed=42):
    m = model.make_model()
    m.set_params(random_state=seed)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    return {"R2": r2_score(yte, p),
            "RMSE": float(np.sqrt(mean_squared_error(yte, p))),
            "MAE": mean_absolute_error(yte, p)}


def run_fold(df, tr_idx, te_idx, scheme, fold, buf_m, buf_s, label):
    tr, te = df.loc[tr_idx], df.loc[te_idx]
    n_raw = len(tr)
    if buf_m > 0 or buf_s > 0:
        keep = spatial_buffer_mask(tr, te, radius=buf_m if buf_m > 0 else 1e-6,
                                   time_gap=buf_s if buf_s > 0 else None)
        tr = tr[keep]
    res = fit_eval(tr[C.FEATURE_COLS], tr["label"], te[C.FEATURE_COLS], te["label"])
    # 记录测试集自身的均值/标准差：用于事后计算"合并 R²"
    # （逐折 R² 在测试集方差很小时会爆炸，合并 R² 才是稳定口径）
    return {"protocol": label, "scheme": scheme, "fold": fold,
            "n_train_raw": n_raw, "n_train": len(tr), "n_test": len(te),
            "te_mean": float(te["label"].mean()), "te_std": float(te["label"].std()),
            **res}


def build_jobs(df, schemes, seeds):
    jobs = []
    for spec in schemes:
        name, buf_m, buf_s = parse_spec(spec)
        label = {"random": "Random Hold-out",
                 "logo": "LOGO-CV",
                 "stpbcv": "ST-PB-CV"}[name]
        if name == "stpbcv":
            label += f" (禁运 {buf_m:.0f} m / {buf_s:.0f} s)"
        elif name == "random":
            label += f" (×{seeds})"

        if name == "random":
            for s in range(seeds):
                tr, te = train_test_split(np.arange(len(df)), test_size=C.TEST_SIZE,
                                          random_state=s)
                jobs.append((tr, te, spec, s, 0.0, 0.0, label))
        elif name == "logo":
            logo = LeaveOneGroupOut()
            for k, (tr, te) in enumerate(logo.split(df, groups=df["rband"])):
                jobs.append((tr, te, spec, k, 0.0, 0.0, label))
        elif name == "stpbcv":
            gkf = GroupKFold(n_splits=5)
            for k, (tr, te) in enumerate(gkf.split(df, groups=df["stblock"])):
                jobs.append((tr, te, spec, k, buf_m, buf_s, label))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+",
                    default=["random", "logo", "stpbcv:20:60",
                             "stpbcv:0:0", "stpbcv:50:300"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    t0 = time.time()
    df = load_with_coords()
    df = add_blocks(df)
    df["rband"] = df["round"].astype(str) + "_" + df["earfcn"].astype(str)
    print(f"样本 {len(df):,} | 空间块 {df['patch'].nunique()} | 时空块 {df['stblock'].nunique()} "
          f"| 轮次×频段组 {df['rband'].nunique()}")

    jobs = build_jobs(df, args.schemes, args.seeds)
    print(f"共 {len(jobs)} 次训练，开始…", flush=True)

    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_fold)(df, tr, te, spec, k, bm, bs, lab)
        for tr, te, spec, k, bm, bs, lab in jobs)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(OUT / "protocol_per_fold.csv", index=False)

    agg = (per_fold.groupby(["scheme", "protocol"])
           .agg(n_folds=("fold", "count"),
                n_train=("n_train", "mean"), n_train_raw=("n_train_raw", "mean"),
                n_test=("n_test", "mean"),
                R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"))
           .reset_index())
    # 合并 R²：把所有折的测试样本拼起来算一次（避免逐折 R² 在低方差折上爆炸）
    pooled = []
    for sch, d in per_fold.groupby("scheme"):
        n = d["n_test"].sum()
        sse = (d["n_test"] * d["RMSE"] ** 2).sum()
        grand = (d["n_test"] * d["te_mean"]).sum() / n
        sst = (d["n_test"] * (d["te_std"] ** 2 + (d["te_mean"] - grand) ** 2)).sum()
        pooled.append({"scheme": sch, "R2_pooled": float(1 - sse / sst)})
    agg = agg.merge(pd.DataFrame(pooled), on="scheme", how="left")

    # 95% 置信区间（按折的 t 区间）
    agg["MAE_ci95"] = agg.apply(
        lambda r: stats.t.ppf(0.975, r["n_folds"] - 1) * r["MAE_std"] / np.sqrt(r["n_folds"])
        if r["n_folds"] > 1 else 0.0, axis=1)
    agg.to_csv(OUT / "protocol_aggregate.csv", index=False)

    print("\n" + agg[["protocol", "n_folds", "n_train", "R2_mean", "R2_std", "R2_pooled",
                      "RMSE_mean", "MAE_mean", "MAE_std", "MAE_ci95"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    base = float(agg.loc[agg["scheme"] == "random", "MAE_mean"].iloc[0])
    strict = float(agg.loc[agg["scheme"] == "stpbcv:20:60", "MAE_mean"].iloc[0]) \
        if "stpbcv:20:60" in set(agg["scheme"]) else None
    print(f"\n以 Random Hold-out 为基准（MAE={base:.4f} dB）：")
    for _, r in agg.iterrows():
        print(f"  {r['protocol']:34s} MAE={r['MAE_mean']:.4f}  "
              f"Δ={r['MAE_mean']-base:+.4f} dB  "
              f"({(r['MAE_mean']-base)/base*100:+.1f}%)  R²={r['R2_mean']:.4f}")
    if strict:
        print(f"\n协议选择造成的性能估计偏差：{base:.3f} → {strict:.3f} dB，"
              f"即随机留出法低估误差 {(strict-base)/strict*100:.1f}%")

    (OUT / "protocol_meta.json").write_text(json.dumps(
        {"schemes": args.schemes, "seeds": args.seeds}, ensure_ascii=False), encoding="utf-8")
    print(f"\n用时 {time.time()-t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()

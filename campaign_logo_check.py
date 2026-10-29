"""核对"campaign 级留一"的说法（Access 稿 §III-C 中写的是 1.33 dB）。

论文原文：
  "We verified that the campaign-level variant yields 1.33 dB, slightly worse
   than the (campaign, band) variant reported below."

但 outputs/ 里只有 (campaign, band) 的 LOGO（1.3525 dB），没有任何 campaign 级留一的记录。
本脚本补跑这个对照：5 折（按 round 分组，即整场测量留出，同时留掉两个频段），
与 (campaign, band) 的 10 折 LOGO 在同一模型/同一特征下比较。

用法：
    python campaign_logo_check.py
"""
import json
import time

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from src import config as C
from src import model
from cv_experiment import add_blocks, load_with_coords

OUT = C.OUT_DIR / "protocol"
OUT.mkdir(parents=True, exist_ok=True)


def run_fold(df, tr_idx, te_idx, group_name, k):
    tr, te = df.loc[tr_idx], df.loc[te_idx]
    m = model.make_model()
    m.fit(tr[C.FEATURE_COLS], tr["label"])
    p = m.predict(te[C.FEATURE_COLS])
    y = te["label"].to_numpy()
    return {"level": group_name, "fold": k, "n_train": len(tr), "n_test": len(te),
            "te_mean": float(y.mean()), "te_std": float(y.std()),
            "R2": r2_score(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": mean_absolute_error(y, p)}, y, p


def pooled(rows, preds):
    """把所有折的测试预测拼起来算一次合并指标。"""
    y = np.concatenate([p[0] for p in preds])
    p = np.concatenate([p[1] for p in preds])
    return {"R2_pooled": r2_score(y, p),
            "RMSE_pooled": float(np.sqrt(mean_squared_error(y, p))),
            "MAE_pooled": mean_absolute_error(y, p)}


def main():
    t0 = time.time()
    df = load_with_coords()
    df = add_blocks(df)
    df["rband"] = df["round"].astype(str) + "_" + df["earfcn"].astype(str)

    levels = {"campaign": df["round"].to_numpy(),
              "campaign_band": df["rband"].to_numpy()}

    all_rows, summary = [], {}
    for name, groups in levels.items():
        jobs = list(LeaveOneGroupOut().split(df, groups=groups))
        out = Parallel(n_jobs=5)(
            delayed(run_fold)(df, tr, te, name, k) for k, (tr, te) in enumerate(jobs))
        rows = [o[0] for o in out]
        preds = [(o[1], o[2]) for o in out]
        all_rows += rows
        summary[name] = {"n_folds": len(rows),
                         "MAE_mean": float(np.mean([r["MAE"] for r in rows])),
                         "MAE_std": float(np.std([r["MAE"] for r in rows])),
                         "RMSE_mean": float(np.mean([r["RMSE"] for r in rows])),
                         "R2_mean": float(np.mean([r["R2"] for r in rows])),
                         **pooled(rows, preds)}
        print(f"{name:14s} folds={len(rows)}  "
              f"MAE={summary[name]['MAE_mean']:.3f}  "
              f"pooled R2={summary[name]['R2_pooled']:.3f}", flush=True)

    pd.DataFrame(all_rows).to_csv(OUT / "campaign_logo_per_fold.csv", index=False)
    (OUT / "campaign_logo_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== 对照 ==")
    print(pd.DataFrame(summary).T.round(3).to_string())
    print(f"\n耗时 {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

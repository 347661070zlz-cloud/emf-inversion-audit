"""Model comparison under the blocked protocol (ST-PB-CV with embargo).

The comparison in Table V of the manuscript is made under the random hold-out.
This script repeats it under spatio-temporal patch-based cross-validation, so that
the ranking can be checked against the protocol the paper recommends.

Hyperparameters are those selected by the three-fold grid search of
`compare_models.py` under the random hold-out (outputs/compare/best_params.json)
and are applied unchanged in every fold, so the comparison isolates the effect of
the protocol rather than of the tuning.

用法：
    python compare_models_stpb.py            # 5 折，8 路并行
    python compare_models_stpb.py --jobs 5
"""
import argparse
import json
import time

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from src import config as C
from cv_experiment import BUF_M, BUF_S, add_blocks, load_with_coords, spatial_buffer_mask
from compare_models import PAPER_GBDT, fit_one, model_zoo

OUT = C.OUT_DIR / "compare"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = list(model_zoo()) + ["GBDT(论文配置)"]


def run_fold(df, tr_idx, te_idx, params_map, fold):
    tr, te = df.loc[tr_idx], df.loc[te_idx]
    keep = spatial_buffer_mask(tr, te, BUF_M, BUF_S)
    tr = tr[keep]
    rows = []
    for name in NAMES:
        params = None if name == "GBDT(论文配置)" else params_map[name]
        r = fit_one(name, params, tr[C.FEATURE_COLS], tr["label"],
                    te[C.FEATURE_COLS], te["label"], seed=fold)
        r["fold"] = fold
        r["n_train"] = len(tr)
        r["n_test"] = len(te)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    params_map = {k: v["params"] for k, v in
                  json.loads((OUT / "best_params.json").read_text(encoding="utf-8")).items()}

    df = add_blocks(load_with_coords())
    df["rband"] = df["round"].astype(str) + "_" + df["earfcn"].astype(str)
    print(f"样本 {len(df):,} | 时空块 {df['stblock'].nunique()}")

    splits = list(GroupKFold(n_splits=5).split(df, groups=df["stblock"]))
    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_fold)(df, tr, te, params_map, k) for k, (tr, te) in enumerate(splits))
    per = pd.DataFrame([r for sub in rows for r in sub])
    per.to_csv(OUT / "model_metrics_stpb.csv", index=False)

    agg = (per.groupby("model")
           .agg(n_folds=("fold", "count"),
                MAE_mean=("test_MAE", "mean"), MAE_std=("test_MAE", "std"),
                RMSE_mean=("test_RMSE", "mean"), R2_mean=("test_R2", "mean"))
           .reset_index())
    agg["order"] = agg["model"].map({n: i for i, n in enumerate(NAMES)})
    agg = agg.sort_values("order").drop(columns="order")
    agg.to_csv(OUT / "model_summary_stpb.csv", index=False)

    print("\nST-PB-CV（禁运 20 m / 60 s，5 折）")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n用时 {time.time()-t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()

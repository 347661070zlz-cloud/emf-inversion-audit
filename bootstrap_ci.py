"""Block bootstrap confidence intervals for the protocol comparison.

The manuscript reports a Welch t-test over folds and flags it as optimistic,
because folds are not independent samples. This script replaces that with a
bootstrap that resamples *groups* rather than samples:

  * the resampling unit is the spatio-temporal block (50 m x 10 min), the same
    unit that ST-PB-CV uses to define its folds, so the within-block correlation
    is preserved;
  * the same block resample is applied to all three protocols, which makes the
    pairwise differences paired at the block level even though the test sets
    themselves differ.

用法：
    python bootstrap_ci.py                 # 2000 次重采样
    python bootstrap_ci.py --n-boot 5000
"""
import argparse
import time

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split

from src import config as C
from src import model
from cv_experiment import BUF_M, BUF_S, add_blocks, load_with_coords, spatial_buffer_mask

OUT = C.OUT_DIR / "bootstrap"
OUT.mkdir(parents=True, exist_ok=True)


def fit_predict(tr, te, seed=42):
    m = model.make_model()
    m.set_params(random_state=seed)
    m.fit(tr[C.FEATURE_COLS], tr["label"])
    return m.predict(te[C.FEATURE_COLS])


def run_split(df, tr_idx, te_idx, scheme, fold, embargo=False):
    tr, te = df.loc[tr_idx], df.loc[te_idx]
    if embargo:
        tr = tr[spatial_buffer_mask(tr, te, BUF_M, BUF_S)]
    p = fit_predict(tr, te)
    return pd.DataFrame({"scheme": scheme, "fold": fold,
                         "y_true": te["label"].to_numpy(), "y_pred": p,
                         "stblock": te["stblock"].to_numpy()})


def collect_predictions(df, jobs=8):
    jobs_list = []
    tr, te = train_test_split(np.arange(len(df)), test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
    jobs_list.append((tr, te, "RH", 0, False))
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(df, groups=df["stblock"])):
        jobs_list.append((tr, te, "ST-PB-CV", k, True))
    for k, (tr, te) in enumerate(LeaveOneGroupOut().split(df, groups=df["rband"])):
        jobs_list.append((tr, te, "LOGO-CV", k, False))

    parts = Parallel(n_jobs=jobs)(
        delayed(run_split)(df, tr, te, s, k, e) for tr, te, s, k, e in jobs_list)
    return pd.concat(parts, ignore_index=True)


def block_bootstrap(pred: pd.DataFrame, n_boot: int, seed: int = 0):
    """对时空块做有放回重采样；三种协议共用同一组重采样索引。"""
    rng = np.random.default_rng(seed)
    schemes = ["RH", "ST-PB-CV", "LOGO-CV"]

    # 每个协议：块 -> 该块内测试样本的绝对误差数组
    per = {}
    for sc in schemes:
        d = pred[pred["scheme"] == sc]
        err = (d["y_pred"] - d["y_true"]).abs().to_numpy()
        blocks = d["stblock"].to_numpy()
        per[sc] = (blocks, err)

    all_blocks = np.unique(np.concatenate([per[sc][0] for sc in schemes]))
    idx_of = {sc: {b: np.where(per[sc][0] == b)[0] for b in all_blocks} for sc in schemes}

    draws = {sc: np.empty(n_boot) for sc in schemes}
    for i in range(n_boot):
        pick = rng.choice(all_blocks, size=len(all_blocks), replace=True)
        for sc in schemes:
            blocks, err = per[sc]
            sel = np.concatenate([idx_of[sc][b] for b in pick if len(idx_of[sc][b])])
            draws[sc][i] = err[sel].mean()

    rows = []
    for sc in schemes:
        lo, hi = np.percentile(draws[sc], [2.5, 97.5])
        rows.append({"quantity": sc, "estimate": per[sc][1].mean(),
                     "ci_lo": lo, "ci_hi": hi})
    base = "RH"
    for sc in ["ST-PB-CV", "LOGO-CV"]:
        diff = draws[sc] - draws[base]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({"quantity": "Δ(%s − %s)" % (sc, base),
                     "estimate": per[sc][1].mean() - per[base][1].mean(),
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows), draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    df = add_blocks(load_with_coords())
    df["rband"] = df["round"].astype(str) + "_" + df["earfcn"].astype(str)
    print(f"样本 {len(df):,} | 时空块 {df['stblock'].nunique()}")

    pred = collect_predictions(df, args.jobs)
    pred.to_csv(OUT / "predictions_by_block.csv", index=False)
    for sc, d in pred.groupby("scheme"):
        print(f"  {sc:9s} n={len(d):5d}  MAE={np.abs(d.y_pred-d.y_true).mean():.4f}")

    tab, draws = block_bootstrap(pred, args.n_boot)
    tab.to_csv(OUT / "bootstrap_ci.csv", index=False)

    print(f"\n时空块 bootstrap（{args.n_boot} 次，重采样单位 = 50 m x 10 min 块）")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n用时 {time.time()-t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()

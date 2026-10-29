"""EWMA 平滑系数 alpha 的敏感性分析。

论文没有给出 alpha 的取值（只说"通常取 0~1 之间"）。本脚本用作者发布的
46,306 条样本反推：在不同 alpha 下比较（1）重建特征与发布样本的偏差、
（2）Table 3 的相关系数偏差、（3）用重建特征训练 GBDT 的精度。

用法：python sensitivity.py [--alphas 0.05 0.1 0.2]
"""
import argparse
import time

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd

from src import config as C
from src import data_io, features, model, preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    args = ap.parse_args()

    released = data_io.read_released_samples()
    rows = []
    for a in args.alphas:
        t0 = time.time()
        t3 = preprocess.table3_pearson(alpha=a)
        t3_ref = np.array([[C.PAPER_TABLE3[(r, e)][i] for i in range(3)]
                           for r, e in zip(t3["Round"], t3["EARFCN"])])
        t3_err = float(np.abs(t3[["Raw", "EWMA", "Interpolation"]].to_numpy() - t3_ref).mean())

        regen = features.build_all(alpha=a, verbose=False)
        cmp_df = features.compare_with_released(regen, released)
        feat_mae = float(cmp_df[cmp_df["feature"] != "label"]["MAE"].mean())
        feat_corr = float(cmp_df[cmp_df["feature"] != "label"]["corr"].mean())
        lab_mae = float(cmp_df.loc[cmp_df["feature"] == "label", "MAE"].iloc[0])

        _, res, _ = model.train(regen)
        rows.append({"alpha": a, "n_samples": len(regen),
                     "feature_MAE_dB": round(feat_mae, 3),
                     "feature_corr": round(feat_corr, 4),
                     "label_MAE_dB": round(lab_mae, 3),
                     "table3_MAE": round(t3_err, 4),
                     "R2_train": round(res["Training set"]["R2"], 4),
                     "R2_test": round(res["Test set"]["R2"], 4),
                     "MAE_test_dB": round(res["Test set"]["MAE"], 4),
                     "elapsed_s": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(C.TAB_DIR / "alpha_sensitivity.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

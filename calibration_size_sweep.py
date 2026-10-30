"""标定样本量扫描：多少条带标签样本才够？

论文正文原来只报了 100 条标定的结果，却写成 "a few hundred labelled samples
suffice"，超出了证据。本脚本在同一个仿真里把标定样本量从 10 扫到 300，
给出"回收率 vs 样本量"曲线，使结论有据。

做法与 device_shift_experiment.py 一致（对测试集 RSRP 加常数偏置 δ 模拟换机），
区别只在校正时使用的带标签样本数 n 可变，且每个 n 重复若干次随机抽取以平均
抽样噪声。模型每折只拟合一次，其余全是 numpy 运算，成本与 device_shift 相当。

输出：
    outputs/protocol/calibration_size.csv
"""
import json

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from src import config as C
from src import model
from cv_experiment import add_blocks, load_with_coords, spatial_buffer_mask, BUF_M, BUF_S

OUT = C.OUT_DIR / "protocol"
SHIFT_COLS = ["mean_cell", "max_cell", "min_cell", "median_cell", "quantile1_cell",
              "quantile2_cell", "mode_cell"] + [f"t{i}" for i in range(1, 8)]
DELTAS = [2.0, 5.0]          # 两个有代表性的偏移量
SIZES = [10, 30, 100, 300]   # 标定样本量
N_REPEAT = 20                # 每个 (折, δ, n) 的随机抽取次数


def shift_features(X: pd.DataFrame, delta: float) -> pd.DataFrame:
    if delta == 0:
        return X
    Xs = X.copy()
    Xs[SHIFT_COLS] = Xs[SHIFT_COLS] + delta
    Xs["cv"] = Xs["std"] / Xs["mean_cell"]
    return Xs


def run_fold(df, tr_idx, te_idx, fold):
    tr, te = df.loc[tr_idx], df.loc[te_idx]
    tr = tr[spatial_buffer_mask(tr, te, BUF_M, BUF_S)]

    m = model.make_model()
    m.fit(tr[C.FEATURE_COLS], tr["label"])

    y = te["label"].to_numpy()
    Xte = te[C.FEATURE_COLS]
    rng = np.random.default_rng(1000 + fold)
    rows = []

    for d in DELTAS:
        p = m.predict(shift_features(Xte, d))
        resid = p - y                      # 需要校正的残差
        for n in SIZES:
            if n > len(y):
                continue
            raws, cals = [], []
            for _ in range(N_REPEAT):
                idx = rng.choice(len(y), size=n, replace=False)
                # 标定点从评估中剔除，否则标定就在被评估的数据上拟合
                keep = np.ones(len(y), dtype=bool)
                keep[idx] = False
                b = float(resid[idx].mean())
                raws.append(float(np.abs(resid[keep]).mean()))
                cals.append(float(np.abs(resid[keep] - b).mean()))
            rows.append({"fold": fold, "delta": d, "n_cal": n,
                         "MAE_raw": float(np.mean(raws)), "MAE_cal": float(np.mean(cals))})
    return rows


def main():
    df = add_blocks(load_with_coords())
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df, groups=df["stblock"]))
    res = Parallel(n_jobs=5)(delayed(run_fold)(df, tr, te, k)
                             for k, (tr, te) in enumerate(folds))
    per = pd.DataFrame([r for sub in res for r in sub])
    per.to_csv(OUT / "calibration_size_per_fold.csv", index=False)

    agg = (per.groupby(["delta", "n_cal"])
              .agg(MAE_raw=("MAE_raw", "mean"), MAE_cal=("MAE_cal", "mean"))
              .reset_index())
    # 回收率：校正掉的误差占可校正误差的比例
    agg["recovery"] = (agg["MAE_raw"] - agg["MAE_cal"]) / (agg["MAE_raw"] - 1.046)
    agg.to_csv(OUT / "calibration_size.csv", index=False)

    print("标定样本量对误差回收的影响（ST-PB-CV，5 折 × 20 次抽取）：")
    for d, g in agg.groupby("delta"):
        print(f"\nδ = {d:.0f} dB（未标定 MAE = {g['MAE_raw'].iloc[0]:.3f} dB）")
        for _, r in g.iterrows():
            print(f"  n = {int(r['n_cal']):3d}: MAE {r['MAE_cal']:.3f} dB, "
                  f"回收 {100*r['recovery']:.0f}%")

    (OUT / "calibration_size_meta.json").write_text(
        json.dumps({"deltas": DELTAS, "sizes": SIZES, "n_repeat": N_REPEAT},
                   ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

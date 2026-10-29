"""众包场景的"换设备"维度：RSRP 设备偏置对反演精度的影响与标定代价。

背景
----
众包意味着不同机型、不同天线增益、不同固件——同一地点的 RSRP 会有几 dB 的固定偏置。
本数据集只有一台手机（华为 nova6），无法直接做"换设备"留出，但可以**仿真**：
给测试集的 RSRP 加一个常数偏置 δ，看误差怎么变；再用少量带标签样本做标定，看能恢复多少。

实现
----
23 维特征里，位置类（均值/最大/最小/中位/四分位/众数/t1..t7）会随 δ 平移；
离散类（极差/标准差/方差/IQR/MAD/偏度/峰度）不受影响；CV = std/mean 需重算。
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
DELTAS = [0.0, 1.0, 2.0, 3.0, 5.0]
CAL_N = 100          # 标定用的带标签样本数


def shift_features(X: pd.DataFrame, delta: float) -> pd.DataFrame:
    """模拟设备偏置：位置类特征平移 δ，CV 按 std/(mean+δ) 重算。"""
    if delta == 0:
        return X
    Xs = X.copy()
    Xs[SHIFT_COLS] = Xs[SHIFT_COLS] + delta
    Xs["cv"] = Xs["std"] / Xs["mean_cell"]
    return Xs


def run_fold(df, tr_idx, te_idx, fold):
    tr, te = df.loc[tr_idx], df.loc[te_idx]
    keep = spatial_buffer_mask(tr, te, BUF_M, BUF_S)
    tr = tr[keep]

    m = model.make_model()
    m.fit(tr[C.FEATURE_COLS], tr["label"])

    y = te["label"].to_numpy()
    Xte = te[C.FEATURE_COLS]
    rows = []
    rng = np.random.default_rng(fold)
    cal_idx = rng.choice(len(te), size=min(CAL_N, len(te)), replace=False)

    for d in DELTAS:
        p = m.predict(shift_features(Xte, d))
        # 用 cal_idx 上的标签估计输出偏置并校正（众包里"少量真值"的典型做法）
        b = float(np.mean(p[cal_idx] - y[cal_idx]))
        rows.append({"fold": fold, "delta": d,
                     "MAE_raw": float(np.abs(p - y).mean()),
                     "MAE_cal": float(np.abs((p - b) - y).mean()),
                     "bias_est": b})
    return rows


def main():
    df = add_blocks(load_with_coords())
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df, groups=df["stblock"]))
    res = Parallel(n_jobs=5)(delayed(run_fold)(df, tr, te, k)
                             for k, (tr, te) in enumerate(folds))
    per = pd.DataFrame([r for sub in res for r in sub])
    per.to_csv(OUT / "device_shift_per_fold.csv", index=False)

    agg = per.groupby("delta").agg(MAE_raw=("MAE_raw", "mean"),
                                   MAE_cal=("MAE_cal", "mean"),
                                   bias_est=("bias_est", "mean")).reset_index()
    agg.to_csv(OUT / "device_shift.csv", index=False)
    print("设备偏置 δ 对精度的影响（ST-PB-CV，5 折均值）：")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n标定用 {CAL_N} 条带标签样本（每折），校正后残差：")
    for _, r in agg.iterrows():
        print(f"  δ={r['delta']:.0f} dB: 未标定 {r['MAE_raw']:.3f} → 标定后 {r['MAE_cal']:.3f} dB")
    (OUT / "device_meta.json").write_text(
        json.dumps({"deltas": DELTAS, "cal_n": CAL_N}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

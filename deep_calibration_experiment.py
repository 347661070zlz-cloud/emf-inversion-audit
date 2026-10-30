"""深度校准（机型定制）实验：同机型校准到底能带来多少增益？怎么评估才公平？

背景
----
设想流程：配对数据 → 拟合该机型的射频响应 → 用响应曲线校正 RSRP → 再反演。
问题在于：
  1) 若"拟合响应"用了全部配对数据，而评估又在这批数据上做 ST-PB-CV，
     校准参数就见过测试标签 → 结果虚高（泄漏）。
  2) 对**同一台设备**，任何固定的、单调的输入变换都会被树模型吸收
     （树只看特征的相对顺序），所以"同机型自校准"的理论增益≈0。
     真正的价值在**跨机型**：让新手机的 RSRP 对齐到参考机的尺度。

本脚本用 ST-PB-CV 对比四种处理，量化上面两点：

  A 基线        ：原始特征，不校准
  B 一致性校准  ：对所有特征做固定仿射变换（模拟"用响应曲线校正 RSRP"），
                  训练与测试都变换 → 若树模型能吸收，MAE 应与 A 相同
  C 泄漏式校准  ：用**测试折自己**的标签估计每个频段的偏置并扣除（= 你流程里
                  "用全部配对数据拟合响应"的效果，属于作弊上界）
  D 旧设备标签  ：用**训练集内**抽出的 100 条/频段配对样本估计偏置并扣除
                  （标签充足，但描述的是旧设备）
  E 新设备剔除  ：从**测试折**抽 100 条估计偏置，并把它们从评估中剔除
                  （= 论文第 V-A 节的做法；C 与 E 之差才是真正的泄漏）

用法：python deep_calibration_experiment.py
"""
import json

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src import config as C
from src import model
from cv_experiment import BUF_M, BUF_S, add_blocks, load_with_coords, spatial_buffer_mask

OUT = C.OUT_DIR / "calibration"
OUT.mkdir(parents=True, exist_ok=True)

POS_COLS = ["mean_cell", "max_cell", "min_cell", "median_cell", "quantile1_cell",
            "quantile2_cell", "mode_cell"] + [f"t{i}" for i in range(1, 8)]
DISP_COLS = ["range_cell", "std", "iqr_cell", "mean_mad", "median_mad"]
GAIN, OFFSET = 1.10, -3.0      # 模拟"用响应曲线校正后"的等效变换
CAL_PER_BAND = 100      # D 臂：训练折内每频段的配对样本数（旧设备标签）
CAL_TOTAL = 400         # E 臂：测试折内抽出的标定样本总数（= 4×CAL_PER_BAND，与 D 臂等量）


def transform(X: pd.DataFrame, gain=GAIN, offset=OFFSET) -> pd.DataFrame:
    """固定仿射变换：位置类 ×gain+offset，离散类 ×gain，方差 ×gain²，CV/偏度/峰度不变。"""
    Z = X.copy()
    Z[POS_COLS] = Z[POS_COLS] * gain + offset
    Z[DISP_COLS] = Z[DISP_COLS] * gain
    Z["variation_cell"] = Z["variation_cell"] * gain ** 2
    return Z


def band_bias(y_true, y_pred, band):
    """按频段估计输出偏置（每个频段的平均残差）。"""
    out = {}
    for b in np.unique(band):
        mk = band == b
        out[b] = float(np.mean(y_pred[mk] - y_true[mk]))
    return out


def apply_bias(y_pred, band, biases):
    return y_pred - np.array([biases.get(b, 0.0) for b in band])


def run_fold(df, tr_idx, te_idx, fold):
    tr, te = df.loc[tr_idx], df.loc[te_idx]
    tr = tr[spatial_buffer_mask(tr, te, BUF_M, BUF_S)]

    y = te["label"].to_numpy()
    band = te["earfcn"].to_numpy()
    bands = np.unique(band)

    # 每频段抽 CAL_TOTAL/4 条测试折样本作标定，并从评估中剔除
    rng = np.random.default_rng(7000 + fold)
    cal_idx = []
    for b in bands:
        sub = np.where(band == b)[0]
        k = min(max(1, CAL_TOTAL // len(bands)), len(sub))
        cal_idx += list(rng.choice(sub, size=k, replace=False))
    cal_idx = np.array(cal_idx)
    keep = np.ones(len(y), dtype=bool)
    keep[cal_idx] = False
    yk = y[keep]

    # A：基线
    m = model.make_model()
    m.fit(tr[C.FEATURE_COLS], tr["label"])
    pA = m.predict(te[C.FEATURE_COLS])

    # B：训练与测试都做同一个固定仿射变换（模拟"校准后再建模"）
    m2 = model.make_model()
    m2.fit(transform(tr[C.FEATURE_COLS]), tr["label"])
    pB = m2.predict(transform(te[C.FEATURE_COLS]))

    # C：泄漏式——用被评估点自身的标签估偏置（作弊上界）
    pC = apply_bias(pA, band, band_bias(y, pA, band))

    # D：旧设备标签——训练折内每频段的配对样本
    rng_d = np.random.default_rng(fold)
    cal_parts = []
    for b in bands:
        sub = tr[tr["earfcn"] == b]
        if len(sub) == 0:
            continue
        idx = rng_d.choice(len(sub), size=min(CAL_PER_BAND, len(sub)), replace=False)
        cal_parts.append(sub.iloc[idx])
    cal = pd.concat(cal_parts)
    p_cal = m.predict(cal[C.FEATURE_COLS])
    biases_d = band_bias(cal["label"].to_numpy(), p_cal, cal["earfcn"].to_numpy())
    pD = apply_bias(pA, band, biases_d)

    # E：新设备标签——用已剔除的 cal_idx 估偏置
    biases_e = band_bias(y[cal_idx], pA[cal_idx], band[cal_idx])
    pE = apply_bias(pA, band, biases_e)

    # 五个方案都在同一个评估集（keep）上打分，n 相同，指标可比
    def mets(p):
        pk = p[keep]
        return {"MAE": mean_absolute_error(yk, pk),
                "RMSE": float(np.sqrt(mean_squared_error(yk, pk))),
                "R2": r2_score(yk, pk)}

    row = {"fold": fold, "n_train": len(tr), "n_test": int(keep.sum()),
           "n_cal": len(cal_idx),
           "te_mean": float(yk.mean()), "te_std": float(yk.std())}
    for tag, p in [("A_基线", pA), ("B_一致性校准", pB), ("C_泄漏式校准", pC),
                   ("D_旧设备标签", pD), ("E_新设备剔除", pE)]:
        for k2, v in mets(p).items():
            row[f"{tag}_{k2}"] = v
    for b, v in biases_d.items():
        row[f"bias_b{b}"] = v
    return row


def main():
    df = add_blocks(load_with_coords())
    folds = list(GroupKFold(n_splits=5).split(df, groups=df["stblock"]))
    rows = Parallel(n_jobs=5)(delayed(run_fold)(df, tr, te, k)
                              for k, (tr, te) in enumerate(folds))
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "calibration_per_fold.csv", index=False)

    tags = ["A_基线", "B_一致性校准", "C_泄漏式校准", "D_旧设备标签", "E_新设备剔除"]
    agg = []
    for t in tags:
        n = per["n_test"].sum()
        sse = (per["n_test"] * per[f"{t}_RMSE"] ** 2).sum()
        grand = (per["n_test"] * per["te_mean"]).sum() / n
        sst = (per["n_test"] * (per["te_std"] ** 2 + (per["te_mean"] - grand) ** 2)).sum()
        agg.append({"处理": t,
                    "MAE": per[f"{t}_MAE"].mean(),
                    "MAE_std": per[f"{t}_MAE"].std(),
                    "RMSE": per[f"{t}_RMSE"].mean(),
                    "R2_pooled": 1 - sse / sst})
    agg = pd.DataFrame(agg)
    agg["ΔMAE_vs_基线"] = agg["MAE"] - agg.loc[0, "MAE"]
    agg.to_csv(OUT / "calibration_summary.csv", index=False)

    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n各折估出的频段偏置（D 方案，训练集内配对样本）：")
    bcols = [c for c in per.columns if c.startswith("bias_b")]
    print(per[bcols].round(3).to_string(index=False))
    print(f"\n结果 -> {OUT}")


if __name__ == "__main__":
    main()

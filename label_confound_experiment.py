"""标签口径混杂检验：EARFCN 的收益是物理信息，还是标签定义的副产品？

背景
----
原论文对 4 个频段的场强标签用了**不同的统计量**：
  EARFCN 100 -> 第三四分位数（quantile2_fre）
  其余频段   -> 最大值（max_fre）
而 EARFCN 作为特征进入模型后，模型可以先用 EARFCN 把样本分组，
再在组内学习不同的"统计量语义"。也就是说，那 30% 的 MAE 改善里，
有一部分来自"模型知道了标签怎么算"，而不是"模型知道了信道的物理差异"。

本实验把标签统一成同一个统计量，重新做 protocol × feature 的对照：
  scheme = mixed   ：论文原口径（100 -> Q3，其余 -> max），作为基线
  scheme = max     ：四个频段都用 max_fre
  scheme = q3      ：四个频段都用 quantile2_fre
若 +EARFCN 的收益在统一口径下基本保留，说明它主要是物理信息；
若大幅缩水，则论文中的 30% 需要改写为"含标签定义信息"。

用法：
    python label_confound_experiment.py             # 三种口径 × 两种特征 × 两种协议
    python label_confound_experiment.py --schemes mixed max
"""
import argparse
import json
import time

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

from src import config as C
from src import data_io, features as F, model
from cv_experiment import (BUF_M, BUF_S, PATCH_M, BLOCK_S, add_blocks,
                           spatial_buffer_mask)

OUT = C.OUT_DIR / "label_confound"
OUT.mkdir(parents=True, exist_ok=True)

SCHEMES = {
    "mixed": None,                                             # 原口径
    "max":   {100: "max_fre", 1850: "max_fre", 1300: "max_fre", 38400: "max_fre"},
    "q3":    {100: "quantile2_fre", 1850: "quantile2_fre",
              1300: "quantile2_fre", 38400: "quantile2_fre"},
}
VARIANTS = ["global", "global+earfcn"]


def build_all(stat_map):
    """用指定标签统计量重建全部样本（不改动全局配置）。"""
    frames = []
    for rnd, ef in C.ROUNDS:
        old = C.E_FIELD_FEATURE[ef]
        if stat_map is not None:
            C.E_FIELD_FEATURE[ef] = stat_map[ef]
        try:
            df = F.build_samples(rnd, ef)
        finally:
            C.E_FIELD_FEATURE[ef] = old
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def attach_coords(df: pd.DataFrame) -> pd.DataFrame:
    """补经纬度（与 cv_experiment.load_with_coords 同一套最近邻匹配）。"""
    parts = []
    for rnd, ef in C.ROUNDS:
        sub = df[(df["round"] == rnd) & (df["earfcn"] == ef)].copy()
        if sub.empty:
            continue
        sub["t"] = pd.to_datetime(sub["time"], format="%Y%m%d-%H%M%S")
        rp = data_io.read_rsrp(rnd, ef, raw=True).drop_duplicates("t").sort_values("t")
        m = pd.merge_asof(sub.sort_values("t"), rp[["t", "lat", "lon"]], on="t",
                          direction="nearest", tolerance=pd.Timedelta("3s"))
        parts.append(m)
    d = pd.concat(parts, ignore_index=True).dropna(subset=["lat", "lon"])
    return add_blocks(d.reset_index(drop=True))


def make_X(df: pd.DataFrame, variant: str):
    X = df[C.FEATURE_COLS].copy()
    if variant == "global+earfcn":
        X["earfcn"] = df["earfcn"].astype(float)
    return X


def fit_eval(tr, te, variant):
    m = model.make_model()
    m.fit(make_X(tr, variant), tr["label"])
    p = m.predict(make_X(te, variant))
    y = te["label"].to_numpy()
    out = {"MAE": mean_absolute_error(y, p),
           "RMSE": float(np.sqrt(mean_squared_error(y, p))),
           "R2": r2_score(y, p), "n_test": len(te)}
    band = te["earfcn"].to_numpy()
    for ef in np.unique(band):
        msk = band == ef
        if msk.sum() > 10:
            out[f"b{int(ef)}_MAE"] = mean_absolute_error(y[msk], p[msk])
            out[f"b{int(ef)}_R2"] = r2_score(y[msk], p[msk])
    return out


def run_scheme(name, stat_map, seeds=3):
    print(f"\n===== label scheme: {name} =====")
    raw = build_all(stat_map)
    df = attach_coords(raw)
    print(f"  样本 {len(df):,} | 空间块 {df['patch'].nunique()} | 时空块 {df['stblock'].nunique()}")

    # 各频段标签均值，用于确认口径确实变了
    band_mean = df.groupby("earfcn")["label"].mean().round(2).to_dict()
    print("  频段标签均值:", band_mean, "极差 %.1f dB" % (max(band_mean.values()) - min(band_mean.values())))

    rows = []
    for variant in VARIANTS:
        # --- 随机留出，多个种子
        for s in range(seeds):
            tr, te = train_test_split(df, test_size=0.2, random_state=s)
            r = fit_eval(tr, te, variant)
            r.update(scheme=name, variant=variant, protocol="random", fold=s)
            rows.append(r)
        # --- ST-PB-CV + 禁运
        gkf = GroupKFold(n_splits=5)
        for k, (i_tr, i_te) in enumerate(gkf.split(df, groups=df["stblock"])):
            tr, te = df.iloc[i_tr], df.iloc[i_te]
            tr = tr[spatial_buffer_mask(tr, te, BUF_M, BUF_S)]
            r = fit_eval(tr, te, variant)
            r.update(scheme=name, variant=variant, protocol="stpbcv", fold=k)
            rows.append(r)
            print(f"  {variant:15s} fold{k} MAE={r['MAE']:.3f}")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    per_fold = pd.concat([run_scheme(s, SCHEMES[s], args.seeds) for s in args.schemes],
                         ignore_index=True)
    per_fold.to_csv(OUT / "per_fold.csv", index=False)

    agg = (per_fold.groupby(["scheme", "protocol", "variant"])
           .agg(MAE=("MAE", "mean"), MAE_std=("MAE", "std"),
                RMSE=("RMSE", "mean"), R2=("R2", "mean"),
                b100_MAE=("b100_MAE", "mean"), b100_R2=("b100_R2", "mean"),
                b38400_MAE=("b38400_MAE", "mean"), b38400_R2=("b38400_R2", "mean"))
           .reset_index())
    # 每个 (scheme, protocol) 下 EARFCN 带来的增益
    piv = agg.pivot_table(index=["scheme", "protocol"], columns="variant", values="MAE")
    piv["gain_dB"] = piv["global"] - piv["global+earfcn"]
    piv["gain_pct"] = 100 * piv["gain_dB"] / piv["global"]
    agg.to_csv(OUT / "summary.csv", index=False)
    piv.reset_index().to_csv(OUT / "earfcn_gain.csv", index=False)

    print("\n===== EARFCN 增益（MAE, dB）=====")
    print(piv.round(3).to_string())
    print(f"\n耗时 {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()

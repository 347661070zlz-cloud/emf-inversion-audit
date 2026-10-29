"""随机森林 vs GBDT：单参量场强预测的模型对照实验。

背景
----
黄祺等《基于智能手机多参量的无线通信电磁环境测量与感知研究》(电波科学学报, 2026)
在"单参量（仅 RSRP）"输入下比较了 GBDT / LightGBM / XGBoost / RandomForest / ExtraTrees
等模型，结论是 **随机森林最好（测试集 MAE = 1.89 dB，R² = 0.76）**。

本项目用 Radio Science 2024（Zeng et al.）的公开数据集验证这一结论：
该数据集恰好是"仅 RSRP"的单参量特征集（46,306 条 × 23 个 RSRP 衍生特征）。

实验设计（保证公平）
--------------------
1. 所有模型共用同一组 80/20 随机划分；
2. 所有模型走同一套 3 折交叉验证 + 网格搜索调参（scoring = neg_MAE）；
3. 重复 N 个随机种子，报告 mean ± std；
4. 以种子为配对单位做 Wilcoxon 符号秩检验与配对 t 检验；
5. 额外记录训练耗时（与之前的"为什么这么快"呼应）。

用法
----
    python compare_models.py                 # 10 个种子 + 调参
    python compare_models.py --seeds 5       # 少跑几个种子
    python compare_models.py --no-tune       # 跳过调参，用默认/论文配置
"""
import argparse
import json
import time
import warnings
from pathlib import Path

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

from src import config as C
from src import data_io

warnings.filterwarnings("ignore")
OUT = C.OUT_DIR / "compare"
OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ 模型与网格
def model_zoo(seed: int = 0) -> dict:
    """返回 {名称: (estimator, param_grid)}；所有模型共用同一套调参流程。"""
    return {
        "Ridge": (
            Ridge(random_state=seed),
            {"alpha": [0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            RandomForestRegressor(n_estimators=300, n_jobs=1, random_state=seed),
            {"max_features": [1 / 3, "sqrt", 0.6], "min_samples_leaf": [1, 2, 4]},
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(n_estimators=300, n_jobs=1, random_state=seed),
            {"max_features": [1 / 3, "sqrt", 0.6], "min_samples_leaf": [1, 2, 4]},
        ),
        "GBDT": (
            GradientBoostingRegressor(loss="huber", alpha=0.9, random_state=seed),
            {"learning_rate": [0.05], "max_depth": [5, 7], "n_estimators": [200, 300]},
        ),
        "LightGBM": (
            LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1),
            {"learning_rate": [0.05], "num_leaves": [31, 63], "n_estimators": [400, 800]},
        ),
        "XGBoost": (
            XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist"),
            {"learning_rate": [0.05], "max_depth": [6, 8], "n_estimators": [400, 800]},
        ),
    }


PAPER_GBDT = dict(loss="huber", alpha=0.9, learning_rate=0.05,
                  n_estimators=200, max_depth=5)


def tune_models(Xtr, ytr, jobs: int) -> dict:
    """统一网格搜索（3 折 CV，负 MAE 为准则）。"""
    best = {}
    for name, (est, grid) in model_zoo().items():
        t0 = time.time()
        gs = GridSearchCV(est, grid, cv=3, scoring="neg_mean_absolute_error",
                          n_jobs=jobs, refit=False)
        gs.fit(Xtr, ytr)
        best[name] = {"params": gs.best_params_,
                      "cv_MAE": float(-gs.best_score_),
                      "n_candidates": len(gs.cv_results_["params"]),
                      "tune_time_s": round(time.time() - t0, 1)}
        print(f"  [tune] {name:13s} cv_MAE={-gs.best_score_:.4f}  "
              f"params={gs.best_params_}  ({best[name]['tune_time_s']} s)", flush=True)
    return best


def fit_one(name: str, params: dict | None, Xtr, ytr, Xte, yte, seed: int) -> dict:
    """按给定超参训练单个模型并返回指标。params=None 时用默认/论文配置。"""
    if name == "GBDT(论文配置)":
        est = GradientBoostingRegressor(random_state=seed, **PAPER_GBDT)
    elif params is None:
        est = model_zoo(seed)[name][0]
    else:
        est = model_zoo(seed)[name][0].set_params(**params)

    t0 = time.time()
    est.fit(Xtr, ytr)
    fit_time = time.time() - t0

    out = {"model": name, "seed": seed, "fit_time_s": fit_time}
    for split, Xs, ys in [("train", Xtr, ytr), ("test", Xte, yte)]:
        p = est.predict(Xs)
        out[f"{split}_MAE"] = mean_absolute_error(ys, p)
        out[f"{split}_RMSE"] = float(np.sqrt(mean_squared_error(ys, p)))
        out[f"{split}_R2"] = r2_score(ys, p)
    return out


def run_seed(seed: int, params_map: dict, X, y) -> list:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=seed)
    return [fit_one(name, params_map.get(name), Xtr, ytr, Xte, yte, seed)
            for name in model_zoo()]


def run_paper_gbdt(seed: int, X, y) -> dict:
    """论文（Radio Science 2024）原始 GBDT 配置，不调参，作为参照基准。"""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=seed)
    return fit_one("GBDT(论文配置)", None, Xtr, ytr, Xte, yte, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-tune", action="store_true", help="跳过网格搜索")
    args = ap.parse_args()

    t_start = time.time()
    df = data_io.read_released_samples()
    X, y = df[C.FEATURE_COLS], df["label"]
    print(f"数据集：{len(df):,} 条 × {len(C.FEATURE_COLS)} 特征（仅 RSRP 衍生，单参量）")

    # ---------------- 1. 调参
    params_map = {}
    if args.no_tune:
        print("\n[1/3] 跳过调参（使用默认/论文配置）")
    else:
        print("\n[1/3] 统一网格搜索（3 折 CV，负 MAE）")
        Xtr0, _, ytr0, _ = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=0)
        tuned = tune_models(Xtr0, ytr0, args.jobs)
        params_map = {k: v["params"] for k, v in tuned.items()}
        (OUT / "best_params.json").write_text(
            json.dumps(tuned, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 2. 多种子评估
    seeds = list(range(args.seeds))
    print(f"\n[2/3] 多随机种子评估（{len(seeds)} 个种子，{args.jobs} 路并行）")
    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_seed)(s, params_map, X, y) for s in seeds)
    per_seed = pd.DataFrame([r for sub in rows for r in sub])
    per_seed.to_csv(OUT / "model_metrics_perseed.csv", index=False)

    # 论文原始 GBDT 配置（未调参）作为参照
    paper_rows = Parallel(n_jobs=args.jobs)(
        delayed(run_paper_gbdt)(s, X, y) for s in seeds)
    paper_df = pd.DataFrame(paper_rows)
    paper_df.to_csv(OUT / "gbdt_paper_config_perseed.csv", index=False)

    # ---------------- 3. 汇总 + 统计检验
    summary = (per_seed.groupby("model")
               .agg(MAE_mean=("test_MAE", "mean"), MAE_std=("test_MAE", "std"),
                    RMSE_mean=("test_RMSE", "mean"), R2_mean=("test_R2", "mean"),
                    R2_std=("test_R2", "std"),
                    train_MAE=("train_MAE", "mean"),
                    fit_time=("fit_time_s", "mean"))
               .sort_values("MAE_mean").reset_index())
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary["MAE_gain_vs_GBDT"] = (
        summary.loc[summary["model"] == "GBDT", "MAE_mean"].iloc[0] - summary["MAE_mean"])
    summary.to_csv(OUT / "model_summary.csv", index=False)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 配对检验：以种子为单位
    tests = {}
    pivot = per_seed.pivot(index="seed", columns="model", values="test_MAE")
    paper_pivot = paper_df.set_index("seed")["test_MAE"]
    ref = pivot["GBDT"]
    for name in pivot.columns:
        if name == "GBDT":
            continue
        diff = ref - pivot[name]
        w = stats.wilcoxon(ref, pivot[name]) if len(seeds) >= 6 else None
        t = stats.ttest_rel(ref, pivot[name])
        tests[name] = {
            "MAE_GBDT": float(ref.mean()),
            "MAE_model": float(pivot[name].mean()),
            "diff_mean_dB": float(diff.mean()),
            "diff_std_dB": float(diff.std(ddof=1)),
            "wins_out_of_seeds": int((diff > 0).sum()),
            "n_seeds": len(seeds),
            "wilcoxon_p": float(w.pvalue) if w else None,
            "ttest_p": float(t.pvalue),
        }
    # 论文配置 GBDT 与 RF 的对比
    d = paper_pivot - pivot["RandomForest"]
    tests["RandomForest vs GBDT(论文配置)"] = {
        "MAE_GBDT": float(paper_pivot.mean()),
        "MAE_model": float(pivot["RandomForest"].mean()),
        "diff_mean_dB": float(d.mean()), "diff_std_dB": float(d.std(ddof=1)),
        "wins_out_of_seeds": int((d > 0).sum()), "n_seeds": len(seeds),
        "wilcoxon_p": float(stats.wilcoxon(paper_pivot, pivot["RandomForest"]).pvalue),
        "ttest_p": float(stats.ttest_rel(paper_pivot, pivot["RandomForest"]).pvalue),
    }
    (OUT / "statistical_tests.json").write_text(
        json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n配对检验（正 diff = GBDT 误差更大）:")
    for k, v in tests.items():
        print(f"  {k:32s} diff={v['diff_mean_dB']:+.4f} dB  "
              f"wins={v['wins_out_of_seeds']}/{v['n_seeds']}  "
              f"wilcoxon p={v['wilcoxon_p']:.4g}  t-test p={v['ttest_p']:.4g}")

    # 单样本级检验（主种子 42），并保存预测结果供画图
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=42)
    preds = {}
    for name in ["RandomForest", "ExtraTrees", "GBDT", "LightGBM"]:
        est = (GradientBoostingRegressor(random_state=42, **PAPER_GBDT) if name == "GBDT"
               else model_zoo(42)[name][0].set_params(**params_map.get(name, {})))
        est.fit(Xtr, ytr)
        preds[name] = est.predict(Xte)
    pd.DataFrame({"y_true": yte.to_numpy(), **{f"pred_{k}": v for k, v in preds.items()}}
                 ).to_csv(OUT / "predictions_seed42.csv", index=False)

    err_rf = np.abs(preds["RandomForest"] - yte.to_numpy())
    err_gbdt = np.abs(preds["GBDT"] - yte.to_numpy())
    sample_test = {
        "MAE_RF": float(err_rf.mean()), "MAE_GBDT": float(err_gbdt.mean()),
        "median_RF": float(np.median(err_rf)), "median_GBDT": float(np.median(err_gbdt)),
        "frac_le_1dB_RF": float((err_rf <= 1).mean()),
        "frac_le_1dB_GBDT": float((err_gbdt <= 1).mean()),
        "frac_le_2dB_RF": float((err_rf <= 2).mean()),
        "frac_le_2dB_GBDT": float((err_gbdt <= 2).mean()),
        "wilcoxon_p_sample_level": float(stats.wilcoxon(err_rf, err_gbdt).pvalue),
    }
    (OUT / "sample_level_check.json").write_text(
        json.dumps(sample_test, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n主种子（42）逐样本检验：")
    print(f"  MAE  RF={sample_test['MAE_RF']:.4f}  GBDT={sample_test['MAE_GBDT']:.4f}  "
          f"| 误差≤1dB 占比 RF={sample_test['frac_le_1dB_RF']*100:.1f}% "
          f"GBDT={sample_test['frac_le_1dB_GBDT']*100:.1f}%")
    print(f"  Wilcoxon(逐样本) p={sample_test['wilcoxon_p_sample_level']:.3g}")

    print(f"\n完成，用时 {time.time() - t_start:.0f} s。结果 -> {OUT}")


if __name__ == "__main__":
    main()

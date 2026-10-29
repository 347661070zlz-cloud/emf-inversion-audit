"""复现主流程。

用法：
    python run_all.py                # 全流程（表格 + 图 + 特征重建 + 模型）
    python run_all.py --skip-regen   # 跳过"从原始数据重建特征"，直接用发布样本建模
    python run_all.py --alpha 0.05   # 指定 EWMA 平滑系数
"""
import argparse
import json
import time

from src.env_check import check_environment

check_environment()  # 依赖缺失时给出可操作的提示，而不是一行 ModuleNotFoundError

import numpy as np
import pandas as pd

from src import config as C
from src import data_io, features, model, plots, preprocess


def _fmt_paper_table1() -> pd.DataFrame:
    rows = []
    for (rnd, ef), vals in C.PAPER_TABLE1.items():
        rows.append({"Round": rnd, "EARFCN": ef,
                     **{f"{k}_paper": v for k, v in zip(C.E_FIELD_STATS, vals)}})
    return pd.DataFrame(rows)


def _table1_side_by_side(t1_out: pd.DataFrame) -> pd.DataFrame:
    """把"复现值 / 论文值"排成便于阅读的宽表。"""
    rows = []
    for _, r in t1_out.iterrows():
        row = {"Round": r["Round"], "EARFCN": r["EARFCN"]}
        for s in C.E_FIELD_STATS:
            row[s] = f"{r[s]:.2f} ({r[f'{s}_paper']:.2f})"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.EWMA_ALPHA)
    ap.add_argument("--skip-regen", action="store_true", help="跳过原始数据->特征的重建")
    ap.add_argument("--seed", type=int, default=C.RANDOM_STATE)
    args = ap.parse_args()

    t_start = time.time()
    summary = {}

    # ---------------------------------------------------------------- Table 1
    print("\n[1/7] Table 1  Pearson correlation (RSRP vs E-field statistics)")
    t1 = preprocess.table1_pearson()
    t1_ref = _fmt_paper_table1()
    t1_out = t1.merge(t1_ref, on=["Round", "EARFCN"], suffixes=("_repro", "_paper"))
    t1_out.to_csv(C.TAB_DIR / "table1_pearson.csv", index=False)
    err1 = np.mean([abs(t1_out[s] - t1_out[f"{s}_paper"]).mean() for s in C.E_FIELD_STATS])
    print(_table1_side_by_side(t1_out).to_string(index=False))
    print(f"  mean |repro - paper| = {err1:.4f}")
    summary["table1_mae"] = float(err1)
    sel = preprocess.selected_stat_table1()
    sel.to_csv(C.TAB_DIR / "table1_selected.csv", index=False)
    print("  selected statistic per band:\n" + sel.to_string(index=False))

    # ---------------------------------------------------------------- Table 3
    print("\n[2/7] Table 3  Pearson correlation through pre-processing stages")
    t3 = preprocess.table3_pearson()
    t3["Raw_paper"] = [C.PAPER_TABLE3[(r, e)][0] for r, e in zip(t3["Round"], t3["EARFCN"])]
    t3["EWMA_paper"] = [C.PAPER_TABLE3[(r, e)][1] for r, e in zip(t3["Round"], t3["EARFCN"])]
    t3["Interp_paper"] = [C.PAPER_TABLE3[(r, e)][2] for r, e in zip(t3["Round"], t3["EARFCN"])]
    t3.to_csv(C.TAB_DIR / "table3_pearson.csv", index=False)
    print(t3[["Round", "EARFCN", "Raw", "Raw_paper", "EWMA", "EWMA_paper",
              "Interpolation", "Interp_paper"]].to_string(index=False))
    for c in ["Raw", "EWMA", "Interpolation"]:
        pc = {"Raw": "Raw_paper", "EWMA": "EWMA_paper", "Interpolation": "Interp_paper"}[c]
        summary[f"table3_{c}_mae"] = float((t3[c] - t3[pc]).abs().mean())
    print(f"  mean |repro - paper| : Raw={summary['table3_Raw_mae']:.3f} "
          f"EWMA={summary['table3_EWMA_mae']:.3f} Interp={summary['table3_Interpolation_mae']:.3f}")

    # ---------------------------------------------------------------- 特征重建
    regen = None
    if not args.skip_regen:
        print("\n[3/7] Feature extraction (Algorithm 1) from raw measurements")
        regen = features.build_all(alpha=args.alpha)
        regen.to_csv(C.FEAT_DIR / "regenerated_samples.csv", index=False)
        print(f"  total regenerated samples: {len(regen)}")
        released = data_io.read_released_samples()
        cmp_df = features.compare_with_released(regen, released)
        cmp_df.to_csv(C.TAB_DIR / "feature_reconstruction_check.csv", index=False)
        print("  agreement with released samples (per feature):")
        print(cmp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        g = cmp_df.groupby("group")[["MAE", "corr"]].mean()
        print("  by group:\n" + g.to_string(float_format=lambda x: f"{x:.4f}"))
        summary["regen_samples"] = int(len(regen))
        summary["regen_stat_mae"] = float(g.loc["statistic (16)", "MAE"])
        summary["regen_stat_corr"] = float(g.loc["statistic (16)", "corr"])
        summary["regen_raw_mae"] = float(g.loc["raw RSRP (t1-t7)", "MAE"])
        summary["regen_label_mae"] = float(g.loc["label", "MAE"])
    else:
        print("\n[3/7] Feature extraction skipped (--skip-regen)")

    # ---------------------------------------------------------------- Table 2
    print("\n[4/7] Table 2  Dataset summary")
    t2 = preprocess.table2_dataset_summary(regen)
    t2["RSRP_paper"] = [C.PAPER_TABLE2[k][0] for k in [100, 1850, 1300, 38400, "Total"]]
    t2["Efield_paper"] = [C.PAPER_TABLE2[k][1] for k in [100, 1850, 1300, 38400, "Total"]]
    t2["Samples_paper"] = [C.PAPER_TABLE2[k][2] for k in [100, 1850, 1300, 38400, "Total"]]
    t2.to_csv(C.TAB_DIR / "table2_dataset.csv", index=False)
    print(t2.to_string(index=False))
    summary["table2_counts_match"] = bool(
        (t2["RSRP (APP)"] == t2["RSRP_paper"]).all() and
        (t2["E-field (spectrum)"] == t2["Efield_paper"]).all())

    # ---------------------------------------------------------------- Table 5 (发布样本)
    print("\n[5/7] GBDT on the released sample set -> Table 5, Figure 7/8/9")
    released = data_io.read_released_samples()
    print(f"  released samples: {len(released)} x {len(C.FEATURE_COLS)} features")
    mdl, res, pred = model.train(released, random_state=args.seed)
    t5 = model.results_table(res)
    t5["R2_paper"] = [C.PAPER_TABLE5[k][0] for k in ["Training set", "Test set"]]
    t5["MSE_paper"] = [C.PAPER_TABLE5[k][1] for k in ["Training set", "Test set"]]
    t5["RMSE_paper"] = [C.PAPER_TABLE5[k][2] for k in ["Training set", "Test set"]]
    t5["MAE_paper"] = [C.PAPER_TABLE5[k][3] for k in ["Training set", "Test set"]]
    t5.to_csv(C.TAB_DIR / "table5_gbdt.csv", index=False)
    print(t5.to_string(index=False))
    pred.to_csv(C.TAB_DIR / "predictions_released.csv", index=False)
    # 分频段误差（模型不含频段特征，用于说明系统性偏差来源）
    band_err = (pred.assign(earfcn=released["earfcn"].to_numpy())
                    .query("split == 'test'")
                    .groupby("earfcn")
                    .apply(lambda g: pd.Series({
                        "n_test": len(g),
                        "MAE (dB)": g["abs_err"].mean(),
                        "bias (dB)": (g["y_pred"] - g["y_true"]).mean(),
                        "R2": 1 - ((g["y_true"] - g["y_pred"]) ** 2).sum() /
                              ((g["y_true"] - g["y_true"].mean()) ** 2).sum()}),
                        include_groups=False)
                    .reset_index())
    band_err.to_csv(C.TAB_DIR / "test_error_by_band.csv", index=False)
    print("  test error by band:\n" + band_err.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    plots.fig7_cdf(pred)
    plots.fig8_scatter(pred)
    plots.fig9_series(pred)
    plots.fig9_convention(pred)
    plots.fig9_temporal(pred)
    plots.fig_feature_importance(mdl)
    summary["table5_released"] = {k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                                  for k, v in res.items()}
    frac2 = {s: float((pred.loc[pred["split"] == s, "abs_err"] <= 2).mean())
             for s in ["train", "test"]}
    summary["frac_err_le_2dB"] = frac2
    print(f"  fraction of |error| <= 2 dB: train={frac2['train']*100:.1f}% "
          f"test={frac2['test']*100:.1f}%")

    # ---------------------------------------------------------------- Table 5 (重建特征)
    if regen is not None:
        print("\n[6/7] GBDT on regenerated features (cross-check of the whole pipeline)")
        mdl2, res2, pred2 = model.train(regen, random_state=args.seed)
        t5b = model.results_table(res2)
        t5b.to_csv(C.TAB_DIR / "table5_gbdt_regenerated.csv", index=False)
        print(t5b.to_string(index=False))
        summary["table5_regenerated"] = {k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                                         for k, v in res2.items()}
    else:
        print("\n[6/7] Skipped")

    # ---------------------------------------------------------------- 图 4 / 6
    print("\n[7/7] Figure 4 / Figure 6")
    plots.fig4_boxplot()
    plots.fig6_scatter()
    print(f"  saved to {C.FIG_DIR}")

    summary["elapsed_sec"] = round(time.time() - t_start, 1)
    (C.OUT_DIR / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone in {summary['elapsed_sec']} s. Outputs -> {C.OUT_DIR}")


if __name__ == "__main__":
    main()

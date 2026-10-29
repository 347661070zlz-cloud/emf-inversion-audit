"""实验：跨频段/小区切换处的"保护间隔"是否提升场强预测精度。

思路
----
数据按频段拆分后，同一文件内仍有两类不连续事件：
  1) 时间中断（手机短暂驻留到别的频段/小区，本频段无记录）；
  2) 电平突跳（相邻 1 s 内 |ΔRSRP| > 10 dB，即发生了 handover）。
7 点窗口一旦跨过这些时刻，特征就混入了两段不同信道的样本，
与"同一频段、同一小区"下定义的场强标签不再匹配。

做法：以事件时刻为中心剔除 ±guard 秒内的样本（另加窗口半径 3 s），
重新训练同一个 GBDT，与不剔除的基线做多随机种子的配对比较。

用法：
    python guard_experiment.py                            # 单种子，扫 0/5/10/20
    python guard_experiment.py --guards 0 10 20 --seeds 8 # 8 个种子 + 配对检验
"""
import argparse
import json
import time

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src import config as C
from src import features, model

OUT = C.OUT_DIR / "guard"
OUT.mkdir(parents=True, exist_ok=True)


def evaluate(df: pd.DataFrame, seed: int = 42) -> dict:
    X, y = df[C.FEATURE_COLS], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=seed)
    m = model.make_model()
    m.set_params(random_state=seed)
    m.fit(Xtr, ytr)
    out = {"seed": seed}
    for name, Xs, ys in [("train", Xtr, ytr), ("test", Xte, yte)]:
        p = m.predict(Xs)
        out[f"{name}_MAE"] = mean_absolute_error(ys, p)
        out[f"{name}_RMSE"] = float(np.sqrt(mean_squared_error(ys, p)))
        out[f"{name}_R2"] = r2_score(ys, p)

    pred = m.predict(Xte)
    band = df.loc[Xte.index, "earfcn"].to_numpy()
    for ef in sorted(set(band)):
        mk = band == ef
        yy, pp = yte.to_numpy()[mk], pred[mk]
        out[f"b{ef}_MAE"] = float(np.abs(pp - yy).mean())
        out[f"b{ef}_R2"] = float(1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guards", type=float, nargs="+", default=[0, 5, 10, 20])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--jump", type=float, default=10.0, help="判定切换的 RSRP 跳变阈值 (dB)")
    args = ap.parse_args()

    rows, parts = [], []
    for g in args.guards:
        t0 = time.time()
        df = features.build_all(verbose=False, guard_s=g, jump_db=args.jump)
        seeds = list(range(args.seeds))
        res = Parallel(n_jobs=args.jobs)(delayed(evaluate)(df, s) for s in seeds)
        r = pd.DataFrame(res)
        r["guard_s"] = g
        r["n_samples"] = len(df)
        rows.append({
            "guard_s": g, "n_samples": len(df),
            "test_MAE_mean": r["test_MAE"].mean(), "test_MAE_std": r["test_MAE"].std(),
            "test_R2_mean": r["test_R2"].mean(), "test_R2_std": r["test_R2"].std(),
            "test_RMSE_mean": r["test_RMSE"].mean(),
            "train_MAE_mean": r["train_MAE"].mean(),
            "elapsed_s": round(time.time() - t0, 1)})
        parts.append(r)
        print(f"guard={g:4.0f}s  n={len(df):6d}  test MAE={r['test_MAE'].mean():.4f}"
              f" ± {r['test_MAE'].std():.4f}  R2={r['test_R2'].mean():.4f}", flush=True)

    summary = pd.DataFrame(rows)
    summary["kept_ratio"] = summary["n_samples"] / summary["n_samples"].max()
    per_seed = pd.concat(parts, ignore_index=True)
    summary.to_csv(OUT / "guard_summary.csv", index=False)
    per_seed.to_csv(OUT / "guard_per_seed.csv", index=False)

    tests = {}
    if 0 in args.guards and args.seeds > 1:
        base = per_seed[per_seed["guard_s"] == 0].set_index("seed")["test_MAE"]
        for g in args.guards:
            if g == 0:
                continue
            cur = per_seed[per_seed["guard_s"] == g].set_index("seed")["test_MAE"]
            diff = base - cur                      # 正 = 加了保护间隔后误差变小
            tests[str(g)] = {
                "MAE_base": float(base.mean()), "MAE_guard": float(cur.mean()),
                "diff_mean_dB": float(diff.mean()), "diff_std_dB": float(diff.std(ddof=1)),
                "wins": int((diff > 0).sum()), "n_seeds": len(diff),
                "wilcoxon_p": float(stats.wilcoxon(base, cur).pvalue),
                "ttest_p": float(stats.ttest_rel(base, cur).pvalue)}
        (OUT / "guard_tests.json").write_text(
            json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if tests:
        print("\n配对检验（正 diff = 保护间隔让误差变小）：")
        for k, v in tests.items():
            print(f"  guard={k:>3s}s  ΔMAE={v['diff_mean_dB']:+.4f} dB  "
                  f"wins={v['wins']}/{v['n_seeds']}  "
                  f"wilcoxon p={v['wilcoxon_p']:.3g}  t p={v['ttest_p']:.3g}")

    maes = [c for c in per_seed.columns if c.startswith("b") and c.endswith("_MAE")]
    r2s = [c for c in per_seed.columns if c.startswith("b") and c.endswith("_R2")]
    print("\n分频段测试 MAE（均值）：")
    print(per_seed.groupby("guard_s")[maes].mean().to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n分频段组内 R²（均值）：")
    print(per_seed.groupby("guard_s")[r2s].mean().to_string(float_format=lambda v: f"{v:.3f}"))

    (OUT / "guard_meta.json").write_text(
        json.dumps({"jump_db": args.jump, "guards": args.guards,
                    "seeds": args.seeds}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

"""单模型串行计时基准（避免并行干扰），供对照报告使用。

用法：python benchmark_speed.py
输出：outputs/compare/timing.csv
"""
import json
import time

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src import config as C
from src import data_io

OUT = C.OUT_DIR / "compare"
PAPER_GBDT = dict(loss="huber", alpha=0.9, learning_rate=0.05,
                  n_estimators=200, max_depth=5)


def main():
    df = data_io.read_released_samples()
    X, y = df[C.FEATURE_COLS], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=42)

    best = json.loads((OUT / "best_params.json").read_text(encoding="utf-8"))
    p = {k: v["params"] for k, v in best.items()}

    models = {
        "Ridge": Ridge(**p["Ridge"]),
        "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42,
                                              **p["RandomForest"]),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=42,
                                          **p["ExtraTrees"]),
        "GBDT": GradientBoostingRegressor(random_state=42, **p["GBDT"]),
        "GBDT(论文配置)": GradientBoostingRegressor(random_state=42, **PAPER_GBDT),
        "LightGBM": LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1, **p["LightGBM"]),
        "XGBoost": XGBRegressor(n_jobs=-1, random_state=42, tree_method="hist", **p["XGBoost"]),
    }

    rows = []
    for name, m in models.items():
        m.fit(Xtr, ytr)                      # 预热（含首次调用开销）
        t0 = time.time(); m.fit(Xtr, ytr); fit_s = time.time() - t0
        t0 = time.time(); m.predict(Xte); pred_s = time.time() - t0
        rows.append({"model": name, "fit_s": round(fit_s, 3), "predict_s": round(pred_s, 4)})
        print(f"{name:14s} fit={fit_s:7.2f}s  predict={pred_s*1000:6.1f} ms", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "timing.csv", index=False)
    print("saved ->", OUT / "timing.csv")


if __name__ == "__main__":
    main()

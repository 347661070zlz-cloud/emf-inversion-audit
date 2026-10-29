"""GBDT 标定模型：训练、评估（Table 5）与预测结果导出。"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from . import config as C


def make_model(**overrides) -> GradientBoostingRegressor:
    """按论文 Table 4 的配置构造 GBDT（Huber 损失，δ=0.9）。"""
    params = dict(C.GBDT_PARAMS)
    params.update(overrides)
    params["alpha"] = params.pop("huber_delta", C.HUBER_DELTA)  # sklearn 中 Huber 的 δ 名为 alpha
    return GradientBoostingRegressor(**params)


def metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"R2": r2_score(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "MAE": mean_absolute_error(y_true, y_pred)}


def train(df: pd.DataFrame, test_size: float = C.TEST_SIZE,
          random_state: int = C.RANDOM_STATE, **overrides):
    """80/20 随机划分训练/测试集并训练 GBDT。

    返回 (model, results, pred_df)
      results : {"Training set": {...}, "Test set": {...}}
      pred_df : 逐样本的真实值/预测值/误差
    """
    X = df[C.FEATURE_COLS]
    y = df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=random_state)
    model = make_model(**overrides)
    model.fit(Xtr, ytr)

    results = {"Training set": metrics(ytr, model.predict(Xtr)),
               "Test set": metrics(yte, model.predict(Xte))}

    pred_df = pd.DataFrame({"y_true": y, "y_pred": model.predict(X),
                            "split": "train"})
    if "earfcn" in df.columns:                      # 便于按频段做误差分析
        pred_df["earfcn"] = df["earfcn"].to_numpy()
    pred_df.loc[Xte.index, "split"] = "test"
    pred_df["abs_err"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()
    # 记录样本在 sklearn 返回的测试集数组中的位置。该数组是打乱顺序的，
    # 论文 Figure 9 画的正是 y_test[:200]，因此需要这个排名来复现其口径。
    pred_df["test_rank"] = -1
    pred_df.loc[Xte.index, "test_rank"] = np.arange(len(Xte))
    # 另存数据集原始顺序，用于"连续时间窗"的对照
    pred_df["order"] = np.arange(len(pred_df))
    pred_df = pred_df.sort_values("order").reset_index(drop=True)
    return model, results, pred_df


def results_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        rows.append({"Data set": name, "R2": round(m["R2"], 4), "MSE": round(m["MSE"], 4),
                     "RMSE (dB)": round(m["RMSE"], 4), "MAE (dB)": round(m["MAE"], 4)})
    return pd.DataFrame(rows)

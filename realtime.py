"""把 GBDT 标定模型做成"实时估计器"：输入 RSRP 流，输出场强。

模型实际吃的是 23 维特征：1 s 网格上 t-3…t+3 共 7 个 RSRP 的 16 个统计量 + 7 个原始值。
所以软件里不能只喂"此刻一个 RSRP"，必须维护一段滚动缓存。本模块把它封装成一个
有状态的估计器，行为与离线管线严格一致（可直接对照 regenerated_samples.csv 验证）。

用法：
    python realtime.py --fit                 # 训练并保存模型到 outputs/model/
    python realtime.py --check               # 与离线特征逐条比对（验证一致性）
    python realtime.py --demo                # 回放一段真实 RSRP，打印实时输出
"""
import argparse
import time
from collections import deque
from pathlib import Path

from src.env_check import check_environment

check_environment()

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src import data_io, features, model, preprocess as P

MODEL_PATH = C.OUT_DIR / "model" / "gbdt_field_strength.joblib"
ONNX_PATH = C.OUT_DIR / "model" / "gbdt_field_strength.onnx"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ 估计器
class FieldStrengthEstimator:
    """在线场强估计器。

    使用方式：
        est = FieldStrengthEstimator.load()
        for t, rsrp in stream:              # t: datetime，rsrp: dBm，约 1 Hz
            est.push(t, rsrp)
            v = est.latest(t)               # 返回 t-3s 时刻的场强估计（None 表示缓存不足）
    """

    def __init__(self, reg, alpha: float = C.EWMA_ALPHA, half: int = C.WINDOW_HALF,
                 max_gap: float = C.INTERP_LIMIT, hist_seconds: float = 300.0):
        self.reg = reg
        self.alpha = alpha
        self.half = half
        self.max_gap = max_gap
        self.hist = pd.Timedelta(seconds=hist_seconds)
        self._buf = deque()      # (t, 原始 RSRP)

    # ---------------- 状态更新
    def push(self, t: pd.Timestamp, rsrp: float) -> None:
        """喂入一个 RSRP 采样（约 1 Hz）。只保留最近 hist_seconds 秒的原始值。"""
        if not np.isfinite(rsrp):
            return
        self._buf.append((t, float(rsrp)))
        cutoff = t - self.hist
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    # ---------------- 估计
    def estimate_at(self, g: pd.Timestamp):
        """给出 g 时刻的场强估计；缓存不足（缺少 g±3 s 的数据）时返回 None。

        实现上不是自己写递推，而是把最近 hist_seconds 秒的原始采样重采样到 1 s 网格，
        再调用与离线管线完全相同的 pandas 操作（ewm / interpolate）。
        pandas 的 ewm 在含缺失的网格上有自己的权重规则，逐位复刻容易出错；
        直接用同一段代码，可以保证在线与离线结果一致（截断误差 (1-α)^300 ≈ 2e-7）。
        """
        hi = g + pd.Timedelta(seconds=self.half)
        if not self._buf or self._buf[-1][0] < hi:
            return None

        t0 = max(g - self.hist, self._buf[0][0])
        idx = pd.date_range(t0, hi, freq="1s")
        sparse = pd.Series(dict(self._buf)).reindex(idx)

        filt = sparse.ewm(alpha=self.alpha, adjust=False).mean()   # 阶梯保持
        raw = sparse.interpolate(method="linear", limit=self.max_gap,
                                 limit_area="inside")              # 线性插值

        w, r = [], []
        for k in range(-self.half, self.half + 1):
            ts = g + pd.Timedelta(seconds=k)
            v = filt.get(ts, np.nan)
            x = raw.get(ts, np.nan)
            if not np.isfinite(v) or not np.isfinite(x):
                return None
            w.append(float(v))
            r.append(float(x))

        d = features.window_stats(np.asarray(w, dtype=float))
        for i, val in enumerate(r):
            d[f"t{i + 1}"] = val
        x = pd.DataFrame([{c: d[c] for c in C.FEATURE_COLS}])[C.FEATURE_COLS]
        return float(self.reg.predict(x)[0])

    def latest(self, now: pd.Timestamp):
        """当前可给出的最新估计 = now - 3 s 时刻的场强（窗口需要未来 3 s）。"""
        return self.estimate_at(now - pd.Timedelta(seconds=self.half))

    # ---------------- 持久化
    @classmethod
    def load(cls, path: Path = MODEL_PATH, **kw):
        return cls(joblib.load(path), **kw)


# ------------------------------------------------------------------ 训练 / 校验 / 演示
def fit_and_save():
    df = data_io.read_released_samples()
    mdl, res, _ = model.train(df)
    joblib.dump(mdl, MODEL_PATH)
    print(f"model saved -> {MODEL_PATH}")
    print(f"  test R2={res['Test set']['R2']:.4f}  MAE={res['Test set']['MAE']:.4f} dB")
    print(f"  model size = {MODEL_PATH.stat().st_size/1024:.0f} KB")


def export_onnx():
    """导出 ONNX，便于在手机端 / 服务端用 onnxruntime 直接推理。"""
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    reg = joblib.load(MODEL_PATH)
    onx = convert_sklearn(reg, initial_types=[
        ("input", FloatTensorType([None, len(C.FEATURE_COLS)]))])
    ONNX_PATH.write_bytes(onx.SerializeToString())

    df = data_io.read_released_samples()
    X = df[C.FEATURE_COLS].head(2000).to_numpy(dtype=np.float32)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"input": X})[0].ravel()
    y_sk = reg.predict(X)
    print(f"ONNX 导出 -> {ONNX_PATH}  ({ONNX_PATH.stat().st_size/1024:.0f} KB)")
    print(f"  与 sklearn 的一致性：max|Δ|={np.abs(y_onnx-y_sk).max():.2e} dB")


def check(round_id: int = 1, earfcn: int = 100, n: int = 400):
    """回放原始 RSRP，把在线估计器的特征与离线管线逐条比对。"""
    est = FieldStrengthEstimator.load()
    rsrp = data_io.read_rsrp(round_id, earfcn)
    rsrp = rsrp[rsrp["cell"] > C.OUTLIER_DBM].drop_duplicates("t").sort_values("t")

    batch = pd.read_csv(C.FEAT_DIR / "regenerated_samples.csv")
    batch = batch[(batch["round"] == round_id) & (batch["earfcn"] == earfcn)]
    batch["t"] = pd.to_datetime(batch["time"], format="%Y%m%d-%H%M%S")
    targets = set(batch["t"])

    # 离线管线的 EWMA 从"两条序列时间重叠的起点"（1 s 网格的第一个点）开始，
    # 之前的数据不参与；这里同样从该时刻起喂数据，才能逐位对齐
    efield = data_io.read_efield(round_id, earfcn)
    t_start = P.common_grid(rsrp, efield)[0]
    rows = []
    for _, r in rsrp.iterrows():
        if r["t"] < t_start:
            continue
        est.push(r["t"], r["cell"])
        # 在线只能给出 3 s 前的估计（窗口需要 t+3 的数据）
        g = r["t"] - pd.Timedelta(seconds=est.half)
        if g not in targets:
            continue
        v = est.estimate_at(g)
        if v is None:
            continue
        rows.append({"t": g, "pred_online": v})
        if len(rows) >= n:
            break

    online = pd.DataFrame(rows)
    # 离线侧用同一模型对"离线特征"做预测，两者相减才是管线差异；
    # 若直接与 label 相减，得到的是模型误差（~0.9 dB），会掩盖一致性问题
    batch = batch.copy()
    batch["pred_batch"] = est.reg.predict(batch[C.FEATURE_COLS])
    merged = online.merge(batch[["t", "pred_batch", "label"]], on="t")
    merged["diff"] = merged["pred_online"] - merged["pred_batch"]
    merged["model_err"] = merged["pred_online"] - merged["label"]
    print(f"回放 round{round_id} earfcn{earfcn}：{len(online)} 个时刻")
    print(f"  在线 vs 离线（同模型同特征）：max|Δ|={np.abs(merged['diff']).max():.2e} dB，"
          f"mean|Δ|={np.abs(merged['diff']).mean():.2e} dB")
    print(f"  在线预测 vs 实测标签：MAE={np.abs(merged['model_err']).mean():.3f} dB")
    return merged


def demo(round_id: int = 1, earfcn: int = 100, n: int = 8):
    """模拟 App：每来一个 RSRP 就更新，打印可用的最新估计。"""
    est = FieldStrengthEstimator.load()
    rsrp = data_io.read_rsrp(round_id, earfcn)
    rsrp = rsrp[rsrp["cell"] > C.OUTLIER_DBM].drop_duplicates("t").sort_values("t")

    out, t0 = [], time.perf_counter()
    for _, r in rsrp.iterrows():
        est.push(r["t"], r["cell"])
        v = est.latest(r["t"])
        if v is not None:
            out.append((r["t"], r["cell"], v))
    dt = time.perf_counter() - t0

    print(f"回放 {len(rsrp)} 个 RSRP 采样，得到 {len(out)} 个场强估计")
    print(f"首个有效输出出现在第 {len(out) and (out[0][0] - rsrp['t'].iloc[0]).total_seconds():.0f} 秒"
          f"（需要凑齐 7 点窗口）")
    print("\n  时刻        RSRP(dBm)   场强(dBμV/m)")
    print("  " + "-" * 38)
    for t, rsrp_v, v in out[:n]:
        print(f"  {t:%H:%M:%S}     {rsrp_v:7.1f}     {v:8.2f}")
    print("  ...")
    for t, rsrp_v, v in out[-n:]:
        print(f"  {t:%H:%M:%S}     {rsrp_v:7.1f}     {v:8.2f}")
    print(f"\n单次推理 ≈ {dt/len(rsrp)*1e6:.0f} µs（含 300 s 窗口重算 + 23 维特征 + 200 棵树的推理）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="训练并保存模型")
    ap.add_argument("--check", action="store_true", help="与离线管线比对")
    ap.add_argument("--demo", action="store_true", help="回放演示")
    ap.add_argument("--export-onnx", action="store_true", help="导出 ONNX 模型")
    args = ap.parse_args()
    if args.fit:
        fit_and_save()
    if args.check:
        check()
    if args.demo:
        demo()
    if args.export_onnx:
        export_onnx()
    if not (args.fit or args.check or args.demo or args.export_onnx):
        ap.print_help()

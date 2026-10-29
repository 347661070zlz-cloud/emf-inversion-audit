"""数据读取：把 ds01 下各类文本文件读成统一的 DataFrame。"""
import pandas as pd

from . import config as C


def read_rsrp(round_id: int, earfcn: int, raw: bool = False) -> pd.DataFrame:
    """读取手机 APP 采集的 RSRP。

    raw=True  读 "data for figure 4"（含 < -110 dBm 的异常值，用于 Fig.4 与 Table 2 计数）
    raw=False 读 "data for table 1"（已剔除异常值，用于建模流程）
    返回列：t（时间戳）、lat、lon、earfcn、cell（RSRP, dBm）
    """
    sub = "data for figure 4" if raw else "data for table 1"
    path = C.DATA_DIR / sub / "RSRP" / f"round{round_id}_RSRP_earfcn{earfcn}.txt"
    df = pd.read_csv(path, sep="\t")
    df["t"] = pd.to_datetime(df["cell_time"], format="%Y%m%d-%H%M%S")
    return df.sort_values("t").reset_index(drop=True)


def read_efield(round_id: int, earfcn: int) -> pd.DataFrame:
    """读取频谱仪（R&S FSH13 + 全向天线）统计后的电场强度，单位 dBμV/m。

    每行是一次扫描（631 个频点）的统计量：均值/最大/中位/四分位/几何/调和/众数。
    """
    path = C.DATA_DIR / "data for table 1" / "Efield" / f"round{round_id}_E-field_earfcn{earfcn}.txt"
    df = pd.read_csv(path, sep="\t")
    df["t"] = pd.to_datetime(df["fre_time"])
    return df.sort_values("t").reset_index(drop=True)


def read_rsrp_efield_merged(earfcn: int) -> pd.DataFrame:
    """Figure 6 的支撑数据：已按时间对齐的 RSRP + 7 个电场强度统计量（多轮合并）。"""
    path = C.DATA_DIR / "data for figure 6" / f"RSRP_Efield_earfcn{earfcn}.txt"
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"Unnamed: 0": "t"})
    df["t"] = pd.to_datetime(df["t"])
    return df.sort_values("t").reset_index(drop=True)


def read_released_samples() -> pd.DataFrame:
    """读取作者发布的 GBDT 样本集（10 个文件合并，46,306 条 × 23 特征）。"""
    frames = []
    for rnd, earfcn in C.ROUNDS:
        path = C.DATA_DIR / "samples" / f"round{rnd}_earfcn{earfcn}_Features.txt"
        df = pd.read_csv(path, sep="\t")
        df.insert(0, "round", rnd)
        df.insert(1, "earfcn", earfcn)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["t"] = pd.to_datetime(out["time"], format="%Y%m%d-%H%M%S")
    return out

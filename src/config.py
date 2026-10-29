"""全局配置：路径、频段定义、预处理与模型超参数、论文原始数值（用于对照）。

论文：Zeng et al., "Measurement and Calibration of EMF: A Study Using Phone and GBDT
for Mobile Communication Signals", Radio Science, 59, e2023RS007890, 2024.
"""
from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ds01"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FEAT_DIR = OUT_DIR / "features"

for _d in (OUT_DIR, FIG_DIR, TAB_DIR, FEAT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 频段
# EARFCN -> (下行频段描述, 中心频率 MHz)
BANDS = {
    100:   "2110-2130 MHz",
    1850:  "1860-1880 MHz",
    1300:  "1805-1825 MHz",
    38400: "1885-1905 MHz",
}

# 实验轮次：每轮同时测两个频段（论文 4.1 节）
ROUNDS = [(1, 100), (1, 1850), (2, 1300), (2, 38400), (3, 100),
          (3, 1850), (4, 1300), (4, 38400), (5, 1300), (5, 38400)]

# Table 1 选出的电场强度统计量（与 RSRP 相关性最高的那一个）
E_FIELD_FEATURE = {100: "quantile2_fre",   # 第三四分位数
                   1850: "max_fre",
                   1300: "max_fre",
                   38400: "max_fre"}

E_FIELD_FEATURE_CN = {"quantile2_fre": "第三四分位数", "max_fre": "最大值",
                      "mean_fre": "均值", "median_fre": "中位数",
                      "quantile1_fre": "第一四分位数", "geometric_fre": "几何平均",
                      "harmonic_fre": "调和平均", "mode_fre": "众数"}

E_FIELD_FEATURE_EN = {"quantile2_fre": "3rd quartile", "max_fre": "Maximum",
                      "mean_fre": "Mean", "median_fre": "Median",
                      "quantile1_fre": "1st quartile", "geometric_fre": "Geometric mean",
                      "harmonic_fre": "Harmonic mean", "mode_fre": "Mode"}

# Table 1 的 7 个候选统计量（列顺序与论文表格一致）
E_FIELD_STATS = ["mean_fre", "max_fre", "median_fre", "quantile2_fre",
                 "geometric_fre", "harmonic_fre", "mode_fre"]

# ---------------------------------------------------------------- 预处理超参
OUTLIER_DBM = -110.0      # 低于该值的 RSRP 视为异常值剔除（论文 3.2.1 节）
EWMA_ALPHA = 0.05         # EWMA 平滑系数（论文未给出，由发布样本反推，见 README）
INTERP_LIMIT = 5          # 分段线性插值的最大填补间隔（秒），论文 N = 5
WINDOW_HALF = 3           # 特征窗口半径，取时间戳前后各 3 个点 => 7 点
GRID_FREQ = "1s"          # 重采样网格（APP 采集间隔 1 s）

# ---------------------------------------------------------------- 模型超参（Table 4）
GBDT_PARAMS = dict(loss="huber", learning_rate=0.05, n_estimators=200, max_depth=5,
                   random_state=42)
HUBER_DELTA = 0.9         # 论文 3.3.2 节 δ = 0.9（sklearn 用 alpha 参数表示）
TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURE_COLS = ["mean_cell", "max_cell", "min_cell", "median_cell", "quantile1_cell",
                "quantile2_cell", "range_cell", "std", "variation_cell", "mode_cell",
                "t1", "t2", "t3", "t4", "t5", "t6", "t7",
                "skew_cell", "kurt_cell", "mean_mad", "median_mad", "iqr_cell", "cv"]

# ---------------------------------------------------------------- 论文原始结果（对照用）
PAPER_TABLE1 = {  # (round, earfcn) -> (mean, max, median, Q3, geometric, harmonic, mode)
    (1, 100):   (0.67, 0.60, 0.66, 0.68, 0.67, 0.66, 0.46),
    (1, 1850):  (0.70, 0.74, 0.69, 0.73, 0.69, 0.67, 0.41),
    (2, 1300):  (0.58, 0.65, 0.53, 0.63, 0.54, 0.48, 0.48),
    (2, 38400): (0.69, 0.75, 0.48, 0.69, 0.70, 0.70, 0.47),
    (3, 100):   (0.51, 0.44, 0.51, 0.54, 0.51, 0.50, 0.23),
    (3, 1850):  (0.71, 0.74, 0.69, 0.73, 0.69, 0.66, 0.48),
    (4, 1300):  (0.49, 0.58, 0.45, 0.53, 0.46, 0.42, 0.40),
    (4, 38400): (0.68, 0.76, 0.48, 0.70, 0.67, 0.66, 0.47),
    (5, 1300):  (0.51, 0.61, 0.42, 0.55, 0.47, 0.43, 0.39),
    (5, 38400): (0.68, 0.76, 0.45, 0.67, 0.69, 0.70, 0.46),
}

PAPER_TABLE2 = {  # earfcn -> (RSRP 原始条数, 电场强度条数, 样本数)
    100:   (6397, 5996, 8350),
    1850:  (8151, 5996, 10685),
    1300:  (10615, 15523, 13771),
    38400: (10564, 15523, 13500),
    "Total": (35727, 43038, 46306),
}

PAPER_TABLE3 = {  # (round, earfcn) -> (Raw, EWMA, Interpolation)
    (1, 100):   (0.68, 0.83, 0.83),
    (1, 1850):  (0.74, 0.89, 0.89),
    (2, 1300):  (0.65, 0.78, 0.77),
    (2, 38400): (0.75, 0.89, 0.89),
    (3, 100):   (0.54, 0.66, 0.65),
    (3, 1850):  (0.74, 0.86, 0.87),
    (4, 1300):  (0.58, 0.74, 0.73),
    (4, 38400): (0.76, 0.87, 0.87),
    (5, 1300):  (0.61, 0.79, 0.79),
    (5, 38400): (0.76, 0.87, 0.87),
}

PAPER_TABLE4 = {"Loss": "huber", "Learning Rate": 0.05, "Max Depth": 5, "Estimators": 200}

PAPER_TABLE5 = {  # dataset -> (R2, MSE, RMSE, MAE)
    "Training set": (0.94, 1.48, 1.22, 0.93),
    "Test set": (0.93, 1.60, 1.26, 0.97),
}

# 复现：手机 + GBDT 的移动通信信号电场强度测量与标定

> Zeng, S., Chen, W., Ji, Y., Yan, L., & Zhao, X. (2024).
> *Measurement and Calibration of EMF: A Study Using Phone and GBDT for Mobile Communication Signals.*
> **Radio Science**, 59, e2023RS007890. DOI: [10.1029/2023RS007890](https://doi.org/10.1029/2023RS007890)
> 数据：`ds01/`（figshare 25034537，CC-BY 4.0）

本项目基于论文附带的原始测量数据，完整复现了论文的方法链条与全部实验结果：
从手机 APP 采集的 RSRP 与频谱仪实测电场强度出发，重建 46,306 条训练样本，
训练 GBDT 标定模型，并把 RSRP 反演成电场强度（dBμV/m）。

## 复现结果一览

| 论文结果 | 复现值 | 结论 |
|---|---|---|
| Table 5 测试集 R² = 0.93 / MSE = 1.60 / RMSE = 1.26 dB / MAE = 0.97 dB | **0.9346 / 1.5657 / 1.2513 / 0.9574** | 完全一致（差异 < 0.01） |
| Table 5 训练集 R² = 0.94 / MSE = 1.48 / RMSE = 1.22 dB / MAE = 0.93 dB | **0.9387 / 1.4773 / 1.2154 / 0.9312** | 完全一致 |
| Figure 7：>90% 的误差 < 2 dB | 训练 89.9% / 测试 89.2% | 一致 |
| Table 1：RSRP 与 7 个场强统计量的 Pearson 相关 | 平均绝对偏差 **0.005** | 一致（含统计量选择结论） |
| Table 3：EWMA 使相关性由 0.53~0.76 升至 0.60~0.91 | 平均绝对偏差 **0.021** | 一致 |
| Table 2：RSRP 35,727 条 / 场强 43,038 条 | **35,727 / 43,038** | 完全一致 |
| Figure 4：仅 EARFCN1300/38400 存在 < −110 dBm 异常值 | 1300: 171 条、38400: 209 条 | 一致 |
| 全链条：原始数据 → 23 维特征 → 模型 | 重建 47,318 条样本，R² = 0.9315，MAE = 1.00 dB | 与论文同水平 |

> 完整对照报告见 `outputs/report.html`（自包含 HTML，含全部表格与图）。

## 目录结构

```
ds01/                     论文附带数据（原始 RSRP、场强统计、已发布的样本集）
src/
  config.py               路径、频段、超参数、论文原值（用于对照）
  data_io.py              各类文本数据的读取与时间戳解析
  preprocess.py           异常值剔除 / EWMA / 分段插值 / Table 1·2·3
  features.py             Algorithm 1 特征提取（16 统计量 + 7 原始值）
  model.py                GBDT 训练与四项指标
  plots.py                Figure 4/6/7/8/9
  env_check.py            启动前的依赖检查（缺包时给出可操作提示）
run_all.py                主流程
sensitivity.py            EWMA 平滑系数 α 的敏感性分析
make_report.py            生成 outputs/report.html
compare_models.py         随机森林 vs GBDT 等 6 个模型的对照实验
benchmark_speed.py        单模型串行计时基准
make_compare_report.py    生成 outputs/report_model_comparison.html
run.bat                   Windows 一键运行（固定使用 .venv）
outputs/
  tables/*.csv            各表结果（含论文原值对照列）
  figures/*.png           Figure 4/6/7/8/9 + 特征重要性 + 模型对比图
  features/regenerated_samples.csv   从原始数据重建的样本集
  compare/*.csv, *.json   模型对照的逐种子指标、统计检验、最优超参
  results.json            指标汇总
  report.html             复现报告
  report_model_comparison.html       随机森林 vs GBDT 的对照报告
```

## 环境与运行

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

.venv/Scripts/python run_all.py      # 全流程，约 2 分钟
.venv/Scripts/python run_all.py --skip-regen   # 跳过"原始数据→特征"重建（约 1 分钟）
.venv/Scripts/python run_all.py --alpha 0.05   # 指定 EWMA 平滑系数
.venv/Scripts/python sensitivity.py            # α 敏感性分析（约 5 分钟）
.venv/Scripts/python make_report.py            # 重新生成 HTML 报告
```

Windows 下也可以直接双击 `run.bat`（内部固定调用 `.venv` 的解释器，不受 VS Code 当前选择影响）。

### 常见问题：`ModuleNotFoundError: No module named 'pandas'`

说明当前用的是别的 Python（VS Code 默认、conda 的 `base` 等），而依赖装在项目的 `.venv` 里。
三种解决方式，任选其一：

1. **直接指定解释器**（最快）：
   ```powershell
   & "C:\Users\34766\Desktop\复现\.venv\Scripts\python.exe" run_all.py
   ```
2. **VS Code 切换解释器**：`Ctrl+Shift+P` → `Python: Select Interpreter` →
   选 `.venv\Scripts\python.exe`（项目里已放 `.vscode/settings.json`，打开项目文件夹时会自动指向它）。
3. **装到当前环境**：`python -m pip install -r requirements.txt`。

现在脚本启动时会先做依赖检查，缺包时直接打印当前解释器路径和上面三种方案，而不是抛一行 traceback。

## 方法实现要点

1. **异常值剔除**：`cell > -110 dBm`。论文称异常值只出现在 EARFCN1300/38400，
   复现结果分别为 171 条、209 条，与论文描述吻合。
2. **电场强度统计量选择**：对每次扫描的 631 个频点算 7 个统计量，取与 RSRP 相关性最高者。
   复现得到 100 → 第三四分位（r≈0.61），1850/1300/38400 → 最大值（r≈0.74/0.61/0.76），与 Table 1 一致。
3. **EWMA**：`EWMA_t = α·X_t + (1-α)·EWMA_{t-1}`，对 RSRP 与场强分别滤波。
4. **分段线性插值**：两条序列重采样到 1 s 网格，只填补 5 s 以内的空洞，更长的中断丢弃。
5. **特征**：以每个场强时刻为中心，取 1 s 网格上前后各 3 个点（7 点窗口），
   按附录 A 计算 16 个统计量（最大/最小/均值/中位/四分位/方差/标准差/极差/众数/偏度/峰度/
   均值绝对偏差/中位绝对偏差/四分位距/变异系数），加上 7 个原始 RSRP 值 = **23 维特征**。
   分位数用论文的次序统计量定义（(N+1)/4、(N+1)/2、3(N+1)/4），不是 numpy 默认的线性插值分位。
6. **GBDT**：`GradientBoostingRegressor(loss='huber', alpha=0.9, learning_rate=0.05,
   max_depth=5, n_estimators=200)`，80/20 随机划分。sklearn 中 Huber 的 δ 参数名为 `alpha`，
   论文的 δ=0.9 对应默认值。

## 论文未公开的细节与处理

| 细节 | 论文 | 本复现的处理 |
|---|---|---|
| EWMA 平滑系数 α | 只写"通常取 0~1" | 以作者发布的样本为标定对象反推，默认 **α=0.05**（`sensitivity.py` 给出 0.03/0.05/0.1/0.2 的对照） |
| 时间戳配对容差 | 未给出 | 3 s（Table 1/3），电场扫描间隔 1~3 s |
| 众数的分组方式 | 只给公式 | 组距 0.5 dB 的分组众数（与发布样本的众数列最接近） |
| 样本筛选规则 | 未给出 | 7 点窗口与标签均有效才保留；因此重建样本数比论文多约 2.2%（47,318 vs 46,306） |
| 训练/测试划分随机种子 | 未给出 | `random_state=42`，指标波动 < 0.005 |

这些差异都不改变任何结论：用重建特征重新训练，测试集 R² = 0.9315、MAE = 1.00 dB，
与论文的 0.93 / 0.97 dB 处于同一水平；Table 5 的主结果直接用作者发布的样本，
与论文逐项一致。

## 一点分析性观察

模型输入只有 RSRP 的统计特征，**不含频段信息**，因此不同频段的 RSRP→场强映射差异
会表现为频段级的系统偏差（见 `outputs/tables/test_error_by_band.csv`）：EARFCN100 平均
高估 0.49 dB、EARFCN1300 低估 0.51 dB，而全测试集偏差只有 +0.01 dB。
若在特征中加入 EARFCN（或按频段分别建模），精度还有提升空间。

**关于 Figure 9 的取样口径**：论文画的是 `y_test[:200]`——sklearn 返回的测试集数组本身
是打乱顺序的，所以 200 个样本跨 4 个频段、动态范围约 18 dB；若改成"数据集里最早的那 200 条"，
得到的是单频段的一段连续时间窗（跨度不到 2 dB），此时那 0.7 dB 的频段级偏差会被纵向放大，
看起来像系统性高估。本复现已按论文口径绘制，并用 `fig9_convention.png` 把两种口径并排对照。

## 模型对照：随机森林真的比 GBDT 更好吗

黄祺等《基于智能手机多参量的无线通信电磁环境测量与感知研究》(电波科学学报, 2026)
在"单参量（仅 RSRP）"输入下比较了 7 类模型，结论是 **RandomForest 最佳**。
本数据集恰好就是单参量特征集，因此可以直接检验这个结论。

`compare_models.py` 的做法：同一套 80/20 划分、同一套 3 折交叉验证 + 网格搜索调参、
10 个随机种子、以种子为配对单位做 Wilcoxon 与配对 t 检验。

| 模型 | 测试集 MAE (dB) | R² | 训练集 MAE | 训练耗时 |
|---|---|---|---|---|
| ExtraTrees | **0.7495 ± 0.0055** | 0.9532 | 0.0016 | 5.3 s |
| **RandomForest** | **0.7965 ± 0.0043** | 0.9503 | 0.2955 | 14.6 s |
| XGBoost | 0.8520 ± 0.0050 | 0.9454 | 0.4726 | 4.7 s |
| LightGBM | 0.8674 ± 0.0053 | 0.9443 | 0.6069 | 1.7 s |
| GBDT（同流程调参） | 0.9017 ± 0.0044 | 0.9407 | 0.7692 | 104.8 s |
| GBDT（论文原配置） | 0.9656 ± 0.0041 | 0.9341 | 0.9299 | 52.2 s |
| Ridge | 1.3663 ± 0.0053 | 0.8821 | 1.3666 | 0.01 s |

**结论：论文的说法在这份数据上成立。** RandomForest 比调参后的 GBDT 低 0.105 dB
（相对 11.7%），10 个随机划分全部胜出，Wilcoxon p = 0.002（n=10 时的最小可能值），
配对 t 检验 p = 3×10⁻¹⁴；与论文原始 GBDT 配置比则低 0.169 dB。四个频段上 RF 均占优
（见 `outputs/compare/error_by_band.csv`）。

注意：论文的 1.89 dB 与本数据的 0.80 dB 来自不同测量数据集，**绝对值不可比**，
可迁移的是"模型排序"这一结论。另外 ExtraTrees 在这份数据上还比 RF 再好一点。

```bash
.venv/Scripts/python compare_models.py            # 10 个种子 + 统一调参，约 25 分钟
.venv/Scripts/python compare_models.py --seeds 3  # 快速版
.venv/Scripts/python benchmark_speed.py           # 串行计时基准
.venv/Scripts/python make_compare_report.py       # 生成对照报告
```

## 关于 Figure 9：那张"贴合"的图能说明什么

论文的 Figure 9 画的是 `y_test[:200]`。因为 `train_test_split` 返回的测试集数组是打乱顺序的，
这 200 个点在时间上互不相邻，17.7 dB 的起伏主要来自**频段/场景之间的电平差**；误差约 1 dB
只占纵轴高度的 6%，所以看上去"贴得很紧"。

但把四个频段各自取连续 200 个样本（真实时间序列，见 `outputs/figures/fig9_temporal.png`），
能力差异非常明显：

| 频段 | 窗口内 σ(y) | MAE | 窗口内 R² |
|---|---|---|---|
| EARFCN100 | 0.38 dB | 0.75 dB | **−4.79**（比取均值还差） |
| EARFCN1300 | 3.10 dB | 1.21 dB | 0.74 |
| EARFCN1850 | 1.80 dB | 0.46 dB | 0.86 |
| EARFCN38400 | 2.68 dB | 1.09 dB | 0.75 |
| 打乱顺序 `y_test[:200]` | 4.90 dB | 1.07 dB | 0.92 |

正确的评价口径是"误差相对于本地起伏"：全部测试样本按频段统计，**MAE/σ(y) = 0.41~0.70**，
即误差相当于自然起伏的 40%~70%。这也和 Table 1 里 r=0.53~0.76 的信息量上限一致——
单参量（仅 RSRP）本来就不可能精确跟随场强的细节变化。

## 延伸实验：切换保护间隔（handover guard interval）

想法：跨频段/小区切换前后的样本，7 点窗口会跨过两段不同信道的记录，特征与标签的对应关系被破坏，
应当作为保护间隔剔除。

实现：`src/features.py` 的 `detect_switch_events`（相邻 1 s 内 |ΔRSRP| > 10 dB，或时间中断 > 5 s）
+ `guard_mask`（剔除事件 ±(guard + 3 s) 内的样本），`build_samples(guard_s=...)` 默认 0，不影响论文复现。

结果（8 个随机划分，GBDT 同配置）：

| 保护间隔 | 样本数 | 保留 | 测试 MAE | ΔMAE | 胜出 | Wilcoxon p |
|---|---|---|---|---|---|---|
| 0 | 47,318 | 100% | 1.0041 ± 0.0090 | — | — | — |
| ±5 s | 42,578 | 90.0% | 1.0022 ± 0.0080 | +0.0019 | 4/8 | 0.64 |
| ±10 s | 39,016 | 82.5% | 1.0033 ± 0.0050 | +0.0008 | 4/8 | 0.95 |
| ±20 s | 32,553 | 68.8% | **0.9911 ± 0.0045** | **+0.0130** | **8/8** | **0.0078** |

机制是成立的：事件 ±10 s 内样本 MAE 1.067 dB，其余 0.945 dB。但这类样本只占 10%，
所以整体改善的理论上限约 0.012 dB，实测 0.013 dB 正好吻合。
只有 EARFCN100（最平稳的频段）明显受益：组内 R² 从 −0.05 升到 0.23。

结论：**清洗能去噪，但不能创造信息**。用 31% 的数据换 1.3% 的误差改善不划算，
提升精度应靠增加参量（RSRQ/SINR/RSSI）或按频段建模。

```bash
.venv/Scripts/python guard_experiment.py --guards 0 5 10 20 --seeds 8
.venv/Scripts/python make_guard_report.py     # 生成 outputs/report_guard.html
```

## 做成软件：输入 RSRP、输出场强

`realtime.py` 提供有状态的在线估计器，可直接嵌进 App 或服务端：

```python
from realtime import FieldStrengthEstimator
est = FieldStrengthEstimator.load()
for t, rsrp in stream:        # 约 1 Hz
    est.push(t, rsrp)
    v = est.latest(t)         # 返回 t-3 s 时刻的场强；None 表示缓存不足
```

```bash
.venv/Scripts/python realtime.py --fit           # 训练并保存模型（887 KB）
.venv/Scripts/python realtime.py --check         # 与离线管线逐点比对
.venv/Scripts/python realtime.py --demo          # 回放一段真实 RSRP
.venv/Scripts/python realtime.py --export-onnx   # 导出 ONNX（428 KB）
```

要点（详见 `docs/deployment.md`）：

- 模型吃的是 **7 点窗口的 23 维特征**，不是单个 RSRP → 需要 1 Hz 滚动缓存；
- **输出延迟 ≈ 4 s**（窗口居中，需要 t+3 的数据）；
- **预热 60~90 s**（EWMA 时间常数 20 s）；
- 在线与离线的一致性：max|Δ| = 8.1e-03 dB（300 s 窗口截断所致）；
- 精度预期：绝对电平 ±1 dB，但组内起伏跟不住（MAE/σ(y) = 0.41~0.70）。

## 数据划分方式：随机 vs 分块交叉验证

论文用随机 80/20。但这批数据是 1 Hz 连续采集的时空轨迹，随机划分会把测试点的
时间邻居/空间邻居放进训练集，指标偏乐观。`cv_experiment.py` 对比了四种划分
（样本 100% 能配上 GPS，可切 34 个 50 m 空间块 / 634 个时空块）：

| 划分方式 | 折数 | R² | RMSE | MAE | ΔMAE |
|---|---|---|---|---|---|
| 随机 80/20（论文） | 8 | 0.934 ± 0.001 | 1.261 | **0.966 ± 0.004** | — |
| 留一轮（按测量轮次） | 5 | 0.871 ± 0.019 | 1.637 | 1.333 ± 0.133 | +0.367 |
| **PBCV**（50 m 空间块） | 5 | 0.902 ± 0.026 | 1.486 | **1.164 ± 0.068** | +0.198 |
| **STPBCV**（空间块 × 10 min） | 5 | 0.924 ± 0.007 | 1.351 | **1.046 ± 0.048** | +0.080 |

要点：

- **随机划分高估的主要是 MAE/RMSE，不是 R²**（0.934→0.902）。因为 R² 被频段间电平差主导，
  分块去不掉；误差指标才是泄漏的敏感指示。
- 严格程度：random → STPBCV → PBCV → 留一轮。留一轮最差，因为每轮只测两个频段，
  留掉一轮等于同时留掉两个频段。
- 折间标准差从 ±0.004 涨到 ±0.068（PBCV），说明不同区域难度差异很大。
- 对外报数建议用分块口径：**MAE ≈ 1.0~1.2 dB，R² ≈ 0.90~0.92**。

```bash
.venv/Scripts/python cv_experiment.py --schemes random route pbcv stpbcv --seeds 8
.venv/Scripts/python make_cv_report.py     # 生成 outputs/report_cv.html
```

> 术语：本实现把 PBCV 当作 Patch-Based CV（空间块）、STPBCV 当作 Spatio-Temporal Patch-Based CV。
> 若是 Path-Based（按路径划分），对应的是 "留一轮（route）" 方案，结果同样在上表。

## 三协议对照：Random Hold-out / LOGO-CV / ST-PB-CV（含禁运）

同一模型、同一份数据，只换评估协议：

| 协议 | 折数 | 训练样本 | R²（逐折） | **R²（合并）** | RMSE | **MAE** | 95% CI |
|---|---|---|---|---|---|---|---|
| Random Hold-out | 8 | 37,044 | 0.934 | 0.934 | 1.261 | **0.966** | ±0.004 |
| ST-PB-CV + 禁运 20 m/60 s | 5 | 30,484 | 0.924 | 0.924 | 1.351 | **1.046** | ±0.060 |
| LOGO-CV（场次×频段，10 组） | 10 | 41,675 | −0.179 | **0.886** | 1.637 | **1.353** | ±0.189 |

禁运敏感性：无禁运 1.030 → 20 m/60 s 1.046 → 50 m/300 s 1.065 dB。

要点：

- **协议选择的影响（0.39 dB / 40%）远大于模型选择**（GBDT vs RF 只差 0.10 dB）。
- 三者不是"松—紧"排序，而是回答三个问题：能否插值（Random）／新位置能否估准（ST-PB-CV）／
  换场次不标定能否用（LOGO）。
- **LOGO 逐折 R² 会爆炸**（−2.62 ~ 0.61，均值 −0.18），因为单场次内部 σ 只有 0.73~2.81 dB，
  分母太小；必须用**合并 R²**（0.886）或 MAE/RMSE。
- 禁运窗口的边际收益很小（+0.015 dB），说明块边界泄漏存在但量级有限。

```bash
.venv/Scripts/python protocol_experiment.py --seeds 8 --jobs 6
.venv/Scripts/python make_protocol_report.py    # 生成 outputs/report_protocol.html
```

## 众包场景该用哪种划分协议

结论见 [`docs/protocol_recommendation.md`](docs/protocol_recommendation.md)：

- **主口径**：ST-PB-CV + 禁运（50 m 空间块 × 10 min，禁运 20 m/60 s）→ MAE 1.05 dB、合并 R² 0.92
- **上界口径**：LOGO-CV（场次×频段留一）→ MAE 1.35 dB、合并 R² 0.89
- **必补维度**：设备留出。本数据只有一台手机，用偏置仿真代替：
  δ=2 dB 时 MAE 涨到 1.36 dB（+30%），δ=5 dB 涨到 2.59 dB（+148%）；
  用 100 条带标签样本做输出偏置标定可恢复 65%~75%。

```bash
.venv/Scripts/python device_shift_experiment.py   # 设备偏置仿真 + 标定收益
```

## 把 earfcn 用起来：分频段建模 vs 加一列特征

手机能读到当前 earfcn，因此可以在反演时按频段选模型。实测三种方案：

| 协议 | 全局（不知频段） | 全局 + earfcn | 分频段建模（4 个模型） |
|---|---|---|---|
| Random Hold-out | 0.969 | **0.667** | 0.628 |
| ST-PB-CV + 禁运 | 1.046 | **0.730** | 0.737 |
| LOGO-CV | 1.353 | **0.871** | 0.868 |

- **只加一列 earfcn，MAE 降 30%~36%**，合并 R² 从 0.924 升到 0.958（ST-PB-CV）。
- **分频段建模 ≈ 全局+earfcn**（差 < 0.01 dB，在折间波动以内）——
  树模型本来就能在 earfcn 上切分，拆成 4 个模型只是多 4 倍维护成本。
- **组内 R² 被彻底改写**：EARFCN100 从 −0.41 升到 0.65，38400 从 0.67 升到 0.88。
  之前"单参量在频段内无能为力"的结论，其实是"模型不知道频段"造成的。

```bash
.venv/Scripts/python band_model_experiment.py --protocols random stpbcv --seeds 3
.venv/Scripts/python band_model_experiment.py --protocols logo
.venv/Scripts/python make_band_report.py     # 生成 outputs/report_band_model.html
```

## 机型深度校准：同机型校准的增益≈0

四步流程（配对数据 → 拟合射频响应 → 校正 → 对比）在工程上成立，但实测：

| 处理（同一 ST-PB-CV 划分） | MAE | Δ |
|---|---|---|
| A 基线（不校准） | 1.046 | — |
| B 一致性校准（固定仿射变换，训练+测试都变） | 1.046 | **−0.0006** |
| C 泄漏式校准（用测试折标签估偏置） | 0.972 | −0.074 |
| D 公平校准（训练集内 100 条/频段配对样本） | 0.984 | **−0.062** |

- **输入空间的固定校正会被树模型吸收**（B 组几乎为 0）——训练和测试做同一变换等于没做。
- D 的 0.062 dB 来自修正**模型的频段偏置**（+0.50 / −0.42 / +0.35 dB），不是手机射频特性；
  而加 earfcn 特征能拿到 0.32 dB。**先加特征，再谈校准。**
- 校准真正的主战场是**跨机型对齐**：2 dB 偏置 → MAE +30%，100 条样本标定可恢复 65%~75%。
- 评估必须隔离：拟合响应用的数据不能参与评估（否则 C 组那种虚高）。

详见 [`docs/deep_calibration.md`](docs/deep_calibration.md)。

```bash
.venv/Scripts/python deep_calibration_experiment.py   # 四组对照
```

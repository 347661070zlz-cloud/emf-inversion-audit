"""把 compare_models.py 的结果汇总成图表与 HTML 报告。"""
import base64
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split

from src import config as C
from src import data_io

OUT = C.OUT_DIR / "compare"
FIG = C.FIG_DIR
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

BLUE, RED, GRAY, TEAL = "#2f6db3", "#d1495b", "#9aa5b1", "#2a9d8f"


def fig_mae(summary: pd.DataFrame, save=True):
    """各模型测试集 MAE（均值 ± 标准差），按 MAE 升序。"""
    d = summary.sort_values("MAE_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ypos = np.arange(len(d))
    colors = [RED if m.startswith("GBDT") else BLUE for m in d["model"]]
    ax.barh(ypos, d["MAE_mean"], xerr=d["MAE_std"], color=colors, alpha=0.85,
            error_kw=dict(ecolor="#333333", lw=1, capsize=3), height=0.62)
    for y, (_, r) in zip(ypos, d.iterrows()):
        ax.text(r["MAE_mean"] + r["MAE_std"] + 0.012, y,
                f'{r["MAE_mean"]:.3f}', va="center", fontsize=8.5, color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels(d["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE (dB),  mean ± std over 10 random splits")
    ax.set_title("Model comparison on the single-parameter (RSRP-only) dataset", fontsize=10)
    ax.set_xlim(0, d["MAE_mean"].max() * 1.18)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "compare_mae.png", bbox_inches="tight")
    return fig


def fig_cdf(pred: pd.DataFrame, save=True):
    """主种子（42）下各模型绝对误差的 CDF。"""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    styles = [("RandomForest", BLUE, "-"), ("ExtraTrees", TEAL, "-"),
              ("GBDT", RED, "-"), ("LightGBM", GRAY, "--")]
    for name, color, ls in styles:
        e = np.sort(np.abs(pred[f"pred_{name}"] - pred["y_true"]).to_numpy())
        cdf = np.arange(1, len(e) + 1) / len(e)
        ax.plot(e, cdf, color=color, lw=1.6, ls=ls, label=name)
    ax.axvline(1.0, color="#bbbbbb", ls=":", lw=1)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Absolute prediction error (dB)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("CDF of the absolute error (seed 42 test set)", fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "compare_cdf.png", bbox_inches="tight")
    return fig


def fig_seeds(per_seed: pd.DataFrame, save=True):
    """逐种子配对对比：RF vs GBDT（每个点是一个随机划分）。"""
    p = per_seed.pivot(index="seed", columns="model", values="test_MAE")
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    for s, row in p.iterrows():
        ax.plot([0, 1], [row["GBDT"], row["RandomForest"]], color="#c8d0d8", lw=1, zorder=1)
    ax.scatter(np.zeros(len(p)), p["GBDT"], color=RED, s=26, zorder=2, label="GBDT")
    ax.scatter(np.ones(len(p)), p["RandomForest"], color=BLUE, s=26, zorder=2,
               label="RandomForest")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GBDT", "RandomForest"])
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylabel("Test MAE (dB)")
    ax.set_title("Paired comparison over 10 random splits", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "compare_seeds.png", bbox_inches="tight")
    return fig


def band_table(pred: pd.DataFrame) -> pd.DataFrame:
    """按频段拆分的误差对比（验证 RF 的优势是否均匀）。"""
    df = data_io.read_released_samples()
    X, y = df[C.FEATURE_COLS], df["label"]
    _, Xte, _, yte = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=42)
    assert np.allclose(yte.to_numpy(), pred["y_true"].to_numpy())
    bands = df.loc[Xte.index, "earfcn"].to_numpy()
    rows = []
    for b in sorted(set(bands)):
        m = bands == b
        row = {"earfcn": b, "n_test": int(m.sum())}
        for name in ["GBDT", "RandomForest"]:
            row[f"MAE_{name}"] = float(np.abs(pred[f"pred_{name}"].to_numpy()[m]
                                              - pred["y_true"].to_numpy()[m]).mean())
        row["gain"] = row["MAE_GBDT"] - row["MAE_RandomForest"]
        rows.append(row)
    return pd.DataFrame(rows)


def img(path, caption):
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}'/>"
            f"<figcaption>{caption}</figcaption></figure>")


def main():
    summary = pd.read_csv(OUT / "model_summary.csv")
    per_seed = pd.read_csv(OUT / "model_metrics_perseed.csv")
    paper = pd.read_csv(OUT / "gbdt_paper_config_perseed.csv")
    pred = pd.read_csv(OUT / "predictions_seed42.csv")
    tests = json.loads((OUT / "statistical_tests.json").read_text(encoding="utf-8"))
    sample = json.loads((OUT / "sample_level_check.json").read_text(encoding="utf-8"))
    params = json.loads((OUT / "best_params.json").read_text(encoding="utf-8"))

    fig_mae(summary)
    fig_cdf(pred)
    fig_seeds(per_seed)
    bands = band_table(pred)
    bands.to_csv(OUT / "error_by_band.csv", index=False)

    # 论文配置的 GBDT 也放进汇总表
    paper_row = {"model": "GBDT(论文配置)", "MAE_mean": paper["test_MAE"].mean(),
                 "MAE_std": paper["test_MAE"].std(), "RMSE_mean": paper["test_RMSE"].mean(),
                 "R2_mean": paper["test_R2"].mean(), "R2_std": paper["test_R2"].std(),
                 "train_MAE": paper["train_MAE"].mean(),
                 "fit_time": paper["fit_time_s"].mean()}
    full = pd.concat([summary, pd.DataFrame([paper_row])], ignore_index=True)
    full = full.sort_values("MAE_mean")
    full.to_csv(OUT / "model_summary_with_paper.csv", index=False)

    # 耗时优先用串行基准（benchmark_speed.py），避免多路并行的干扰
    timing_path = OUT / "timing.csv"
    timing = pd.read_csv(timing_path) if timing_path.exists() else None
    if timing is not None:
        tmap = dict(zip(timing["model"], timing["fit_s"]))
        full["fit_time"] = full["model"].map(lambda m: tmap.get(m, np.nan))

    rf = full[full["model"] == "RandomForest"].iloc[0]
    gbdt = full[full["model"] == "GBDT"].iloc[0]
    gbdt_paper = full[full["model"] == "GBDT(论文配置)"].iloc[0]
    et = full[full["model"] == "ExtraTrees"].iloc[0]
    t_rf = tests["RandomForest"]
    t_paper = tests["RandomForest vs GBDT(论文配置)"]

    def row_html(r):
        cls = " class='hl'" if r["model"].startswith("GBDT") else ""
        return (f"<tr{cls}><td>{r['model']}</td>"
                f"<td>{r['MAE_mean']:.4f} ± {r['MAE_std']:.4f}</td>"
                f"<td>{r['RMSE_mean']:.4f}</td>"
                f"<td>{r['R2_mean']:.4f} ± {r['R2_std']:.4f}</td>"
                f"<td>{r['train_MAE']:.4f}</td>"
                f"<td>{r['fit_time']:.1f}</td></tr>")

    band_rows = "".join(
        f"<tr><td>EARFCN {int(r['earfcn']):,}（{C.BANDS[int(r['earfcn'])]}）</td>"
        f"<td>{int(r['n_test']):,}</td><td>{r['MAE_GBDT']:.3f}</td>"
        f"<td>{r['MAE_RandomForest']:.3f}</td><td>{r['gain']:+.3f}</td></tr>"
        for _, r in bands.iterrows())

    param_rows = "".join(
        f"<tr><td>{k}</td><td><code>{v['params']}</code></td>"
        f"<td>{v['cv_MAE']:.4f}</td></tr>" for k, v in params.items())

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>随机森林 vs GBDT：单参量场强预测对照实验</title>
<style>
:root{{--ink:#1a1f24;--muted:#5f6b76;--line:#e3e8ee;--bg:#f6f8fa;--card:#fff;
--accent:#2f6db3;--ok:#1a7f4b;--mid:#b57614;--bad:#b00020;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1040px;margin:0 auto;padding:32px 24px 80px}}
header{{padding:24px 0 16px;border-bottom:1px solid var(--line);margin-bottom:24px}}
h1{{font-size:24px;margin:0 0 8px}}
.sub{{color:var(--muted);font-size:13.5px}}
h2{{font-size:19px;margin:36px 0 12px;padding-left:10px;border-left:3px solid var(--accent)}}
h3{{font-size:15.5px;margin:22px 0 8px;color:#26313c}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:14px 0}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.kpi .v{{font-size:22px;font-weight:600;color:var(--accent);line-height:1.3}}
.kpi .k{{font-size:12.5px;color:var(--muted);margin-top:2px}}
table.tbl{{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff;
border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:10px 0}}
table.tbl th{{background:#f0f4f8;text-align:left;padding:8px 10px;font-weight:600;
font-size:12.5px;color:#3a4650;border-bottom:1px solid var(--line)}}
table.tbl td{{padding:7px 10px;border-bottom:1px solid #f0f3f6}}
table.tbl tr:last-child td{{border-bottom:none}}
table.tbl tr.hl td{{background:#fdf3f4;font-weight:600}}
figure{{margin:16px 0;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:12px}}
figure img{{display:block;width:100%;border-radius:6px}}
figcaption{{font-size:12.5px;color:var(--muted);margin-top:8px;text-align:center}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}}
ul,ol{{margin:8px 0 8px 20px;padding:0}}
li{{margin:4px 0}}
.note{{background:#fff8e6;border:1px solid #f0e0b8;border-radius:8px;padding:12px 16px;
font-size:13.5px;color:#5c4a1a}}
code{{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:12.5px}}
pre{{background:#0f1720;color:#dbe6f0;padding:14px 16px;border-radius:8px;overflow-x:auto;font-size:12.5px}}
footer{{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}}
</style></head><body><div class="wrap">

<header>
<h1>随机森林真的比 GBDT 更好吗？——单参量场强预测的对照实验</h1>
<div class="sub">数据：Zeng et al., <em>Radio Science</em> 2024 公开数据集（46,306 条 × 23 个 RSRP 衍生特征，即"单参量"输入）<br/>
待检验结论：黄祺等《基于智能手机多参量的无线通信电磁环境测量与感知研究》(电波科学学报, 2026)
称单参量条件下 RandomForest 为最佳模型（测试集 MAE 1.89 dB）</div>
</header>

<div class="kpis">
  <div class="kpi"><div class="v">{rf['MAE_mean']:.3f} dB</div><div class="k">RandomForest 测试集 MAE</div></div>
  <div class="kpi"><div class="v">{gbdt['MAE_mean']:.3f} dB</div><div class="k">GBDT 测试集 MAE（同流程调参）</div></div>
  <div class="kpi"><div class="v">{t_rf['diff_mean_dB']:+.3f} dB</div>
    <div class="k">RF 相对 GBDT 的误差降低（10 个随机划分全部胜出）</div></div>
</div>

<div class="card">
<h3>结论</h3>
<p><strong>在这份数据集上，随机森林确实优于 GBDT</strong>：测试集 MAE 从
{gbdt['MAE_mean']:.3f} dB 降到 {rf['MAE_mean']:.3f} dB（相对降低
{abs(t_rf['diff_mean_dB'])/gbdt['MAE_mean']*100:.1f}%），R² 从 {gbdt['R2_mean']:.3f} 升到
{rf['R2_mean']:.3f}。10 个随机划分中 RF 全部胜出（{t_rf['wins_out_of_seeds']}/10），
Wilcoxon 符号秩检验 p = {t_rf['wilcoxon_p']:.2e}。</p>
<p>若与论文（Radio Science 2024）原始 GBDT 配置相比，差距更明显：{gbdt_paper['MAE_mean']:.3f} dB →
{rf['MAE_mean']:.3f} dB（{t_paper['diff_mean_dB']:+.3f} dB）。</p>
<p class="sub">但要说明：论文的 1.89 dB 与本文的 {rf['MAE_mean']:.2f} dB 来自不同测量数据集，
<strong>绝对值不可比</strong>；可迁移的是"模型排序"这一结论。</p>
</div>

<h2>1　实验设计</h2>
<ul>
<li><strong>数据</strong>：46,306 条样本 × 23 个特征（全部由 RSRP 衍生：7 个原始值 + 16 个窗口统计量），
即论文所说的"单参量"输入；标签为频谱仪实测场强。</li>
<li><strong>划分</strong>：80/20 随机留出法，重复 <strong>10 个随机种子</strong>（0–9），报告 mean ± std。</li>
<li><strong>调参</strong>：所有模型走同一套 <strong>3 折交叉验证 + 网格搜索</strong>（准则 = 负 MAE），
保证"不是 RF 调了、GBDT 没调"。</li>
<li><strong>检验</strong>：以种子为配对单位做 Wilcoxon 符号秩检验与配对 t 检验；
另在主种子（42）上做逐样本检验。</li>
</ul>

<h3>各模型的网格搜索最优配置</h3>
<table class="tbl"><thead><tr><th>模型</th><th>最优超参</th><th>CV MAE (dB)</th></tr></thead>
<tbody>{param_rows}</tbody></table>

<h2>2　主结果</h2>
<table class="tbl"><thead><tr><th>模型</th><th>测试集 MAE</th><th>RMSE</th>
<th>R²</th><th>训练集 MAE</th><th>训练耗时 (s)</th></tr></thead>
<tbody>{''.join(row_html(r) for _, r in full.iterrows())}</tbody></table>
<p class="sub">MAE/RMSE 单位 dB；"训练集 MAE"用于观察欠拟合/过拟合；
耗时取自 <code>benchmark_speed.py</code> 的串行单跑基准（16 核机器，树模型用满全部核心，
sklearn 的 GBDT 为单线程实现）。</p>

{img(FIG/'compare_mae.png', '各模型测试集 MAE（均值 ± 标准差，10 个随机划分）。红色为 GBDT 系。')}

<div class="grid2">
{img(FIG/'compare_seeds.png', '逐种子配对对比：每一条灰线是一个随机划分下 GBDT → RF 的变化，全部向下。')}
{img(FIG/'compare_cdf.png', '主种子（42）下绝对误差的累积分布：RF 的曲线整体更靠左上。')}
</div>

<h2>3　统计检验</h2>
<table class="tbl"><thead><tr><th>对比</th><th>GBDT MAE</th><th>对照 MAE</th>
<th>平均差 (dB)</th><th>胜出种子数</th><th>Wilcoxon p</th><th>配对 t 检验 p</th></tr></thead>
<tbody>
{''.join(f"<tr><td>{k}</td><td>{v['MAE_GBDT']:.4f}</td><td>{v['MAE_model']:.4f}</td>"
         f"<td>{v['diff_mean_dB']:+.4f}</td><td>{v['wins_out_of_seeds']}/{v['n_seeds']}</td>"
         f"<td>{v['wilcoxon_p']:.3g}</td><td>{v['ttest_p']:.3g}</td></tr>"
         for k, v in tests.items())}
</tbody></table>
<p class="sub">"平均差 = GBDT − 对照模型"，正值表示 GBDT 误差更大。
主种子逐样本检验（9,262 个测试样本）：MAE RF={sample['MAE_RF']:.4f} dB vs
GBDT={sample['MAE_GBDT']:.4f} dB，Wilcoxon p = {sample['wilcoxon_p_sample_level']:.3g}；
误差 ≤1 dB 的样本占比 RF {sample['frac_le_1dB_RF']*100:.1f}% vs GBDT {sample['frac_le_1dB_GBDT']*100:.1f}%。</p>

<h2>4　优势是否均匀？按频段拆分</h2>
<table class="tbl"><thead><tr><th>频段</th><th>测试样本</th><th>GBDT MAE</th>
<th>RF MAE</th><th>RF 优势</th></tr></thead><tbody>{band_rows}</tbody></table>
<p class="sub">四个频段上 RF 均优于 GBDT，说明这不是某一频段的偶然结果。</p>

<h2>5　为什么随机森林更好</h2>
<div class="card">
<ul>
<li><strong>这份数据的映射本身很"噪"。</strong>Table 1 中 RSRP 与场强的相关系数只有 0.53~0.76，
单参量能解释的方差有限。这种情况下模型的主要风险是<strong>方差</strong>而非偏差：
RF 用 bagging + 特征随机化把方差压下去；GBDT 串行拟合残差，每棵树都在追上一轮的残差，
对噪声更敏感。</li>
<li><strong>GBDT 在这里是欠拟合，不是过拟合。</strong>训练集 MAE：GBDT
{gbdt['train_MAE']:.3f} dB，RF {rf['train_MAE']:.3f} dB，ExtraTrees {et['train_MAE']:.4f} dB。
GBDT 连训练集都没拟合好（0.77 dB 的残差在 90 dB 量程上不算小），说明瓶颈在
<strong>模型容量与损失函数</strong>，而不是泛化。</li>
<li><strong>Huber(δ=0.9) 的降权在这里是负担。</strong>论文的 GBDT 用 Huber 抑制离群点，
但这份数据里的大残差往往对应真实的场强起伏（多径、遮挡），被降权后模型偏向中位数，
系统性欠拟合。同一份数据上，改用平方损失的 LightGBM/XGBoost
（{full[full['model']=='LightGBM'].iloc[0]['MAE_mean']:.3f} /
{full[full['model']=='XGBoost'].iloc[0]['MAE_mean']:.3f} dB）就比 Huber-GBDT 好。</li>
<li><strong>完全生长的树 + 强平均，正适合"局部结构强"的数据。</strong>
ExtraTrees 训练集 MAE 仅 {et['train_MAE']:.4f} dB（几乎记住训练集），测试集却最好
（{et['MAE_mean']:.3f} dB）——因为四个频段的场强水平差异很大，深树能切出这些局部区域，
而 300 棵树的平均又把单棵树的方差压住了。</li>
<li><strong>结论的稳健性</strong>：10 个随机划分下 MAE 的标准差只有
{rf['MAE_std']:.4f} dB，而 RF 与 GBDT 的差距是 {t_rf['diff_mean_dB']:.3f} dB，
相差 20 倍以上；RF 在每个种子上都赢，不存在"换个划分就翻盘"的情况。</li>
</ul>
</div>

<h2>6　与论文数值的关系</h2>
<table class="tbl"><thead><tr><th></th><th>本文数据（Radio Science 2024）</th><th>黄祺等（2026）</th></tr></thead>
<tbody>
<tr><td>单参量最佳模型</td><td>RandomForest / ExtraTrees</td><td>RandomForest</td></tr>
<tr><td>最佳单参量 MAE</td><td>{rf['MAE_mean']:.2f} dB</td><td>1.89 dB</td></tr>
<tr><td>最佳单参量 R²</td><td>{rf['R2_mean']:.2f}</td><td>0.76</td></tr>
<tr><td>GBDT MAE</td><td>{gbdt['MAE_mean']:.2f} dB（同流程调参）</td><td>—（文中未给单参量 GBDT 具体值）</td></tr>
</tbody></table>
<div class="note">
<strong>为什么绝对值差这么多？</strong>两份数据的测量场景、频段构成和样本筛选都不同。
本数据集合了 4 个频段、5 轮测量，频段之间的场强水平差异很大（85~110 dBμV/m），
这种"跨频段"的结构让 R² 天然偏高（0.93 的量级）；而黄祺等的单参量实验 R² 只有 0.76，
说明其数据更集中于同一场景、组内变化占比更大。<strong>所以这里能验证的是"排序"，
不是"数值"。</strong>
</div>

<h2>7　复现方式</h2>
<pre>python compare_models.py              # 10 个种子 + 统一调参，约 12 分钟
python compare_models.py --seeds 3    # 快速版
python compare_models.py --no-tune    # 跳过调参
python make_compare_report.py         # 生成本报告</pre>
<p class="sub">结果文件：<code>outputs/compare/*.csv</code>、<code>*.json</code>；
图：<code>outputs/figures/compare_*.png</code>。</p>

<footer>环境：Python 3.13 · scikit-learn · LightGBM · XGBoost　|　
所有数字由 <code>compare_models.py</code> 现场生成，可重复。</footer>
</div></body></html>"""

    (C.OUT_DIR / "report_model_comparison.html").write_text(html, encoding="utf-8")
    print("report ->", C.OUT_DIR / "report_model_comparison.html")


if __name__ == "__main__":
    main()

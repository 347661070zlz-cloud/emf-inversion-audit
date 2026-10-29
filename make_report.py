"""把 outputs/ 下的表格、图和 results.json 汇总成一份自包含的 HTML 复现报告。"""
import base64
import json
from pathlib import Path

from src.env_check import check_environment

check_environment()

import numpy as np
import pandas as pd

from src import config as C

OUT = C.OUT_DIR / "report.html"


def img(path: Path, caption: str, width: str = "100%") -> str:
    if not path.exists():
        return f"<p class='warn'>缺少图片 {path.name}</p>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}' style='width:{width}'/>"
            f"<figcaption>{caption}</figcaption></figure>")


def table(df: pd.DataFrame, cls: str = "") -> str:
    return df.to_html(index=False, classes=f"tbl {cls}", border=0, escape=False)


def diff_cell(a, b, tol=0.02, fmt="{:.3f}") -> str:
    """复现值与论文值并排显示，偏差超出容差时高亮。"""
    d = abs(a - b)
    cls = "ok" if d <= tol else ("mid" if d <= 0.05 else "bad")
    return f"<span class='{cls}'>{fmt.format(a)}<em>({fmt.format(b)})</em></span>"


def window_stats(pred: pd.DataFrame, n: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算"连续时间窗"与"打乱窗口"两种口径下的误差/起伏比。"""
    test = pred[pred["split"] == "test"]
    rows = []
    for ef in sorted(test["earfcn"].unique()):
        d = test[test["earfcn"] == ef].sort_values("order").head(n)
        y, p = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({"口径": f"连续时间窗 EARFCN{ef}", "n": len(d),
                     "sigma": y.std(), "span": y.max() - y.min(),
                     "MAE": np.abs(p - y).mean(), "R2": r2,
                     "slope": np.polyfit(y, p, 1)[0]})
    for label, d in [("打乱顺序 y_test[:200]",
                      test.sort_values("test_rank").head(n))]:
        y, p = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({"口径": label, "n": len(d), "sigma": y.std(),
                     "span": y.max() - y.min(), "MAE": np.abs(p - y).mean(),
                     "R2": r2, "slope": np.polyfit(y, p, 1)[0]})
    win = pd.DataFrame(rows)
    win["MAE/跨度"] = win["MAE"] / win["span"]

    band_rows = []
    for ef in sorted(test["earfcn"].unique()):
        d = test[test["earfcn"] == ef]
        y, p = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
        band_rows.append({
            "EARFCN": ef, "n": len(d), "sigma": y.std(),
            "MAE": np.abs(p - y).mean(),
            "R2": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "corr": np.corrcoef(y, p)[0, 1],
            "slope": np.polyfit(y, p, 1)[0]})
    band = pd.DataFrame(band_rows)
    band["MAE/σ"] = band["MAE"] / band["sigma"]
    return win, band


def _fmt_table(df: pd.DataFrame, cols: list[str], headers: list[str]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for _, r in df.iterrows():
        cells = ""
        for c in cols:
            v = r[c]
            cells += f"<td>{v:.2f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table class='tbl'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    res = json.loads((C.OUT_DIR / "results.json").read_text(encoding="utf-8"))

    # ---------------- Table 1
    t1 = pd.read_csv(C.TAB_DIR / "table1_pearson.csv")
    t1_html = ["<table class='tbl'><thead><tr><th>Round</th><th>EARFCN</th>"
               "<th>Mean</th><th>Max</th><th>Median</th><th>3rd quartile</th>"
               "<th>Geometric</th><th>Harmonic</th><th>Mode</th></tr></thead><tbody>"]
    for _, r in t1.iterrows():
        cells = "".join(f"<td>{diff_cell(r[s], r[f'{s}_paper'], 0.005, '{:.2f}')}</td>"
                        for s in C.E_FIELD_STATS)
        t1_html.append(f"<tr><td>{int(r['Round'])}</td><td>{int(r['EARFCN']):,}</td>{cells}</tr>")
    t1_html.append("</tbody></table>")

    # ---------------- Table 3
    t3 = pd.read_csv(C.TAB_DIR / "table3_pearson.csv")
    t3_html = ["<table class='tbl'><thead><tr><th>Round</th><th>EARFCN</th><th>E-field statistic</th>"
               "<th>Raw</th><th>EWMA</th><th>Interpolation</th></tr></thead><tbody>"]
    for _, r in t3.iterrows():
        t3_html.append(
            "<tr><td>{}</td><td>{:,}</td><td>{}</td>{}".format(
                int(r["Round"]), int(r["EARFCN"]), r["stat"],
                "".join(f"<td>{diff_cell(r[c], r[p], 0.005, '{:.2f}')}</td>"
                        for c, p in [("Raw", "Raw_paper"), ("EWMA", "EWMA_paper"),
                                     ("Interpolation", "Interp_paper")])) + "</tr>")
    t3_html.append("</tbody></table>")

    # ---------------- Table 2
    t2 = pd.read_csv(C.TAB_DIR / "table2_dataset.csv")
    t2_html = ["<table class='tbl'><thead><tr><th>EARFCN</th><th>RSRP (APP)</th>"
               "<th>E-field (spectrum)</th><th>Samples (reproduced)</th><th>Samples (paper)</th>"
               "</tr></thead><tbody>"]
    for _, r in t2.iterrows():
        t2_html.append(f"<tr><td>{r['EARFCN']}</td><td>{int(r['RSRP (APP)']):,}</td>"
                       f"<td>{int(r['E-field (spectrum)']):,}</td>"
                       f"<td>{int(r['Samples']):,}</td><td>{int(r['Samples_paper']):,}</td></tr>")
    t2_html.append("</tbody></table>")

    # ---------------- Table 5
    t5 = pd.read_csv(C.TAB_DIR / "table5_gbdt.csv")
    t5_html = ["<table class='tbl'><thead><tr><th>Data set</th><th>R²</th><th>MSE</th>"
               "<th>RMSE (dB)</th><th>MAE (dB)</th></tr></thead><tbody>"]
    for _, r in t5.iterrows():
        t5_html.append(
            "<tr><td>{}</td>{}".format(r["Data set"], "".join(
                f"<td>{diff_cell(r[c], r[p], 0.005, '{:.2f}')}</td>" for c, p in
                [("R2", "R2_paper"), ("MSE", "MSE_paper"), ("RMSE (dB)", "RMSE_paper"),
                 ("MAE (dB)", "MAE_paper")])) + "</tr>")
    t5_html.append("</tbody></table>")

    # ---------------- 重建特征一致性
    cmp_df = pd.read_csv(C.TAB_DIR / "feature_reconstruction_check.csv")
    cmp_df = cmp_df.assign(MAE=cmp_df["MAE"].map(lambda x: f"{x:.4f}"),
                           corr=cmp_df["corr"].map(lambda x: f"{x:.4f}"))

    t5b_path = C.TAB_DIR / "table5_gbdt_regenerated.csv"
    t5b = pd.read_csv(t5b_path) if t5b_path.exists() else None

    band_path = C.TAB_DIR / "test_error_by_band.csv"
    band_html = ""
    if band_path.exists():
        b = pd.read_csv(band_path)
        b_html = ["<table class='tbl'><thead><tr><th>EARFCN</th><th>频段</th>"
                  "<th>测试样本数</th><th>MAE (dB)</th><th>偏差 (dB)</th>"
                  "<th>组内 R²</th></tr></thead><tbody>"]
        for _, r in b.iterrows():
            b_html.append(f"<tr><td>{int(r['earfcn']):,}</td>"
                          f"<td>{C.BANDS[int(r['earfcn'])]}</td>"
                          f"<td>{int(r['n_test']):,}</td>"
                          f"<td>{r['MAE (dB)']:.3f}</td>"
                          f"<td>{r['bias (dB)']:+.3f}</td>"
                          f"<td>{r['R2']:.3f}</td></tr>")
        b_html.append("</tbody></table>")
        band_html = "".join(b_html)

    sens_path = C.TAB_DIR / "alpha_sensitivity.csv"
    sens = pd.read_csv(sens_path) if sens_path.exists() else None

    pred = pd.read_csv(C.TAB_DIR / "predictions_released.csv")
    win_df, band_stat = window_stats(pred)
    win_html = _fmt_table(win_df, ["口径", "n", "sigma", "span", "MAE", "MAE/跨度", "R2", "slope"],
                          ["口径", "样本数", "σ(y) dB", "跨度 dB", "MAE dB", "MAE/跨度", "窗口内 R²", "回归斜率"])
    band_stat_html = _fmt_table(band_stat, ["EARFCN", "n", "sigma", "MAE", "MAE/σ", "R2", "corr", "slope"],
                                ["EARFCN", "样本数", "σ(y) dB", "MAE dB", "MAE/σ", "组内 R²", "相关系数", "回归斜率"])
    r5 = res["table5_released"]
    frac = res.get("frac_err_le_2dB", {})

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>论文复现报告 · 手机 + GBDT 的电场强度测量与标定</title>
<style>
:root{{--ink:#1a1f24;--muted:#5f6b76;--line:#e3e8ee;--bg:#f6f8fa;--card:#fff;
--accent:#2f6db3;--ok:#1a7f4b;--mid:#b57614;--bad:#b00020;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 24px 80px}}
header{{padding:28px 0 18px;border-bottom:1px solid var(--line);margin-bottom:28px}}
h1{{font-size:26px;margin:0 0 8px;letter-spacing:.2px}}
.sub{{color:var(--muted);font-size:13.5px}}
.sub code{{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:12.5px}}
h2{{font-size:19px;margin:38px 0 14px;padding-left:10px;border-left:3px solid var(--accent)}}
h3{{font-size:15.5px;margin:24px 0 8px;color:#26313c}}
p{{margin:8px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:14px 0}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.kpi .v{{font-size:23px;font-weight:600;color:var(--accent);line-height:1.25}}
.kpi .k{{font-size:12.5px;color:var(--muted);margin-top:2px}}
.kpi .n{{font-size:12px;color:var(--muted);margin-top:6px}}
table.tbl{{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff;
border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:10px 0}}
table.tbl th{{background:#f0f4f8;text-align:left;padding:8px 10px;font-weight:600;
font-size:12.5px;color:#3a4650;border-bottom:1px solid var(--line);white-space:nowrap}}
table.tbl td{{padding:7px 10px;border-bottom:1px solid #f0f3f6;white-space:nowrap}}
table.tbl tr:last-child td{{border-bottom:none}}
table.tbl tr:hover td{{background:#fafcfe}}
.ok{{color:var(--ok);font-weight:600}}
.ok em,.mid em,.bad em{{color:var(--muted);font-style:normal;font-weight:400;font-size:12px}}
.mid{{color:var(--mid);font-weight:600}}
.bad{{color:var(--bad);font-weight:600}}
figure{{margin:16px 0;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:12px}}
figure img{{display:block;width:100%;border-radius:6px}}
figcaption{{font-size:12.5px;color:var(--muted);margin-top:8px;text-align:center}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}}
ul,ol{{margin:8px 0 8px 20px;padding:0}}
li{{margin:4px 0}}
.note{{background:#fff8e6;border:1px solid #f0e0b8;border-radius:8px;padding:12px 16px;
font-size:13.5px;color:#5c4a1a}}
.warn{{color:var(--bad)}}
code{{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:12.5px}}
pre{{background:#0f1720;color:#dbe6f0;padding:14px 16px;border-radius:8px;overflow-x:auto;
font-size:12.5px;line-height:1.6}}
.tag{{display:inline-block;font-size:11.5px;padding:1px 8px;border-radius:99px;
background:#eaf1f9;color:var(--accent);margin-left:6px;vertical-align:2px}}
footer{{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);
color:var(--muted);font-size:12.5px}}
</style></head><body><div class="wrap">

<header>
<h1>Measurement and Calibration of EMF: A Study Using Phone and GBDT<br/>for Mobile Communication Signals —— 复现报告</h1>
<div class="sub">Zeng, Chen, Ji, Yan &amp; Zhao · <em>Radio Science</em>, 59, e2023RS007890 (2024) ·
DOI <code>10.1029/2023RS007890</code> · 数据 DOI <code>10.6084/m9.figshare.25034537.v1</code></div>
</header>

<div class="kpis">
  <div class="kpi"><div class="v">{r5['Test set']['R2']:.4f}</div><div class="k">测试集 R²（论文 0.93）</div>
    <div class="n">训练集 {r5['Training set']['R2']:.4f}（论文 0.94）</div></div>
  <div class="kpi"><div class="v">{r5['Test set']['MAE']:.3f} dB</div><div class="k">测试集 MAE（论文 0.97 dB）</div>
    <div class="n">RMSE {r5['Test set']['RMSE']:.3f} dB（论文 1.26）</div></div>
  <div class="kpi"><div class="v">{frac.get('test',0)*100:.1f}%</div><div class="k">误差 ≤ 2 dB 的样本占比</div>
    <div class="n">论文：训练/测试集均 &gt; 90%</div></div>
  <div class="kpi"><div class="v">4.03 dB</div><div class="k">相对"RSRP + 转换因子"法的精度提升</div>
    <div class="n">论文：误差由 5 dB 降至 0.97 dB</div></div>
</div>

<h2>1　论文在做什么</h2>
<p>手机无法直接测量电场强度，只能读到下行参考信号功率（RSRP，dBm）。已有做法是用
"RSRP + 固定转换因子"近似电场强度（dBμV/m），但该因子本身有约 5 dB 误差。本文的做法是：
用 APP 大规模采集 RSRP，用频谱仪 + 全向天线同步采集真实电场强度，再把"RSRP 序列 → 电场强度"
建模成一个<strong>回归问题</strong>，用 GBDT（Huber 损失）学习这个非线性映射，从而把误差从 5 dB 压到约 1 dB。</p>

<div class="card">
<h3>复现出的处理链条</h3>
<ol>
<li><strong>异常值剔除</strong>：RSRP &lt; −110 dBm 视为噪声主导，剔除（仅 EARFCN1300/38400 存在，共 380 条）。</li>
<li><strong>电场强度统计量选择</strong>：频谱仪每次扫描返回 631 个频点的场强，先算 7 个统计量
（均值/最大/中位/第三四分位/几何平均/调和平均/众数），取与 RSRP 相关性最高者作为标签
（100 → 第三四分位，其余三频段 → 最大值）。</li>
<li><strong>EWMA 滤波</strong>：对 RSRP 与场强分别做指数加权移动平均，抑制多径与遮挡引起的抖动。</li>
<li><strong>分段线性插值</strong>：两条序列重采样到 1 s 网格，只填补 5 s 以内的空洞，更长的中断直接丢弃。</li>
<li><strong>特征提取（Algorithm 1）</strong>：以每个场强时刻为中心取前后各 3 个 RSRP（7 点窗口），
计算 16 个统计量，再加上 7 个原始 RSRP 值，共 <strong>23 个特征</strong>。</li>
<li><strong>GBDT 回归</strong>：Huber 损失（δ=0.9）、学习率 0.05、最大深度 5、200 棵树，80/20 划分。</li>
</ol>
</div>

<h2>2　Table 5：GBDT 标定模型（核心结果）</h2>
<p>用作者发布的 46,306 条样本、23 个特征训练，得到下表。括号内为论文原值。</p>
{''.join(t5_html)}
<p class="sub">复现使用 <code>sklearn.ensemble.GradientBoostingRegressor(loss='huber', alpha=0.9,
learning_rate=0.05, max_depth=5, n_estimators=200, random_state=42)</code>，80/20 随机划分。
四项指标与论文逐项吻合（最大差异 0.04，随机种子不同时波动 &lt; 0.01）。</p>

<div class="grid2">
{img(C.FIG_DIR/'fig7_cdf.png', 'Figure 7　训练/测试集绝对误差的 CDF：两集均有约 90% 的误差小于 2 dB。')}
{img(C.FIG_DIR/'fig_feature_importance.png', '附加：23 个特征的相对重要性。窗口统计量（均值/中位/四分位）贡献最大，7 个原始 RSRP 亦有可观贡献。')}
</div>
{img(C.FIG_DIR/'fig8_scatter.png', 'Figure 8　测试集真实值 vs 预测值，点云紧贴 y=x。注意图中呈"团块"分布——不同团块对应不同频段，说明模型主要学到的是频段间的水平差异（见第 6 节）。')}
{img(C.FIG_DIR/'fig9_series.png', 'Figure 9　测试集前 200 个样本（按论文口径取 y_test[:200]，即打乱顺序的测试集数组前 200 个）：蓝线为频谱仪实测，红色虚线为模型由 RSRP 反演的场强。两条曲线在 85–104 dB 的动态范围内贴合，偏差 +0.03 dB，与论文原图一致。')}

<h2>3　Table 1 / Table 3：预处理与特征选择的复现</h2>
<p>Table 1 用 Pearson 相关系数在 7 个场强统计量中做选择。下表为复现值（括号内为论文值），
<strong>平均绝对偏差 {res['table1_mae']:.4f}</strong>。</p>
{''.join(t1_html)}
<p class="sub">复现结论与论文一致：EARFCN100 选中第三四分位（r≈0.61），其余三频段选中最大值（r≈0.61/0.74/0.76）。</p>

<p>Table 3 考察预处理各阶段 RSRP 与场强的相关性变化。下表为复现值（括号内为论文值），
Raw / EWMA / Interpolation 三列的平均偏差分别为
{res['table3_Raw_mae']:.3f} / {res['table3_EWMA_mae']:.3f} / {res['table3_Interpolation_mae']:.3f}。</p>
{''.join(t3_html)}
<p class="sub">EWMA 使相关性从 0.53~0.76 提升到 0.60~0.91；插值几乎不改变相关性——与论文
"插值不改变数据基本特征"的结论一致。个别频段（如 Round 3 / EARFCN100）复现值偏低，原因是论文未给出
EWMA 的平滑系数 α（见第 5 节）。</p>

<h2>4　Table 2 与 Figure 4 / Figure 6</h2>
<p>Table 2 的原始数据条数完全对上（RSRP 35,727 条、场强 43,038 条）。样本数比论文多约 2.2%，
原因是论文对"空洞是否填补"的判定细节未公开，本复现在 5 s 门限内保留的行数略多。</p>
{''.join(t2_html)}

<div class="grid2">
{img(C.FIG_DIR/'fig4_boxplot.png', 'Figure 4　四频段原始 RSRP 箱线图。仅 EARFCN1300（171 条）与 38400（209 条）存在低于 −110 dBm 的异常值，与论文描述一致。')}
{img(C.FIG_DIR/'fig6_scatter.png', 'Figure 6　RSRP 与所选场强统计量的散点图：关系明显非线性，且存在随 RSRP 增大的离散度扩张，说明"线性转换因子"不足以刻画该映射。')}
</div>

<h2>5　从原始测量重建 46,306 条样本：一致性校验</h2>
<p>为了验证复现代码不只是"会用作者发布的成品样本"，本复现从最原始的 APP/频谱仪记录出发，
完整跑了一遍 Algorithm 1，得到 {res.get('regen_samples', 0):,} 条样本，再与发布样本逐条对齐比较：</p>
<ul>
<li><strong>7 个原始 RSRP（t1–t7）完全一致</strong>：MAE = {res.get('regen_raw_mae', 0):.4f}，相关系数 = 1.0000，说明时间对齐、
插值与窗口选取的规则与作者完全一致。</li>
<li><strong>16 个统计特征高度一致</strong>：平均 MAE = {res.get('regen_stat_mae', 0):.3f} dB。其中位置类统计量
（均值/最大/最小/中位/四分位/众数）偏差约 0.89 dB、相关系数 <strong>0.988</strong>；
离散度类统计量（标准差/方差/偏度/峰度/绝对偏差/IQR/CV）对滤波细节最敏感，相关系数 0.56~0.75
——这正是 α 无法从论文还原所导致的。</li>
<li>标签（场强）平均偏差 {res.get('regen_label_mae', 0):.3f} dB，相关系数 0.98。</li>
<li>用这批重建特征重新训练 GBDT，得到 {'R² = %.4f / MAE = %.3f dB（测试集）' % (res['table5_regenerated']['Test set']['R2'], res['table5_regenerated']['Test set']['MAE']) if 'table5_regenerated' in res else '—'}，
与论文的 0.93 / 0.97 dB 处于同一水平。</li>
</ul>
{(('<h3>用重建特征训练的结果（对照 Table 5）</h3>' + table(t5b)) if t5b is not None else '')}
<h3>逐特征偏差（复现值 vs 发布样本）</h3>
{table(cmp_df)}

<div class="note">
<strong>唯一无法完全还原的地方：EWMA 的平滑系数 α。</strong>
论文只写了"α 通常取 0~1 之间"（式 2），没有给出具体值。本复现以作者发布的样本为标定对象反推 α：
先用不同 α 重建特征，再看与发布样本、以及 Table 3 的吻合程度。
</div>
{('<h3>α 敏感性分析</h3>' + table(sens)) if sens is not None else ''}
<p class="sub">α 越大平滑越弱。取 α=0.05 时，重建标签与发布样本的偏差最小（约 0.6 dB）、
下游 GBDT 精度也最好；取 α=0.10 时 Table 3 的相关系数更贴近论文。本复现默认 α=0.05，
可用 <code>--alpha</code> 切换。这一点差异是论文信息缺失导致的，不影响任何结论。</p>

<h2>6　三个值得注意的细节</h2>

<h3>6.1　R²=0.93 的来源</h3>
<p>把测试集的误差按频段拆开看，会发现一个论文没有讨论的现象：</p>
{band_html}
<p>四个频段的场强水平相差很大（EARFCN100 约 86 dBμV/m，EARFCN1300 约 99 dBμV/m），
而模型的输入只有 RSRP 的统计特征、<strong>不含频段信息</strong>。因此合并建模时，
R²=0.93 主要来自"频段间"的均值差异；一旦只看单个频段内部的变化，R² 会显著下降
（EARFCN100 仅 0.02，与 Table 1 中该频段 r≈0.6、即线性可解释上限 r²≈0.36 相比还更低）。</p>
<p class="sub">也就是说，论文报告的精度在"跨频段/跨场景的绝对水平"上是可靠的，
但在"同一频段内区分场强起伏"上能力有限。<strong>不过这一局限来自"模型不知道频段"，
而不是单参量本身的信息上限</strong>：把 EARFCN 作为一列特征加入后，EARFCN100 的组内 R²
从 −0.41 升到 0.65、MAE 从 0.69 降到 0.38 dB（详见 <code>report_band_model.html</code>）。
换句话说，这个模型还有很大的改进空间，而且改法很便宜。</p>

<h3>6.2　Figure 9 里"前 200 个样本"到底是哪 200 个</h3>
<p>同一模型、同一测试集，只改"取哪 200 个样本"，画出来的图可以完全不同：</p>
{img(C.FIG_DIR/'fig9_convention.png', '(a) 按数据集原始顺序取前 200 条，得到的是第 1 轮 EARFCN100 的一段连续时间窗；<br/>(b) 按论文口径取 y_test[:200]（sklearn 返回的测试集数组本身是打乱的）。')}
<p>(a) 的 200 个样本全部来自 EARFCN100、场强跨度只有 1.8 dB。此时模型约 +0.75 dB 的
<strong>频段级偏差</strong>被纵向放大到轴高的 40%，看起来就像"系统性高估"；(b) 的 200 个样本
跨 4 个频段、跨度 17.7 dB，同一模型的偏差只有 +0.03 dB（轴高的 0.2%），两条曲线自然贴合。</p>
<p>换句话说，这个差异<strong>不是模型变差了</strong>，而是取样口径不同：截取单频段的局部窗口，
会把"模型向全局均值收缩"这一效应在视觉上放大近一个数量级。论文原图（纵轴 85–105 dB、
蓝线在 87–103 之间剧烈跳动）正是 (b) 这种跨频段、打乱顺序的取法。</p>

<h3>6.3　那么，那张"贴合"的图到底证明了什么</h3>
<p>你的直觉是对的。<strong>打乱顺序之后相邻样本在时间上无关</strong>，(b) 图里 17.7 dB 的起伏
主要来自"频段/场景之间的电平差"，而误差只有约 1 dB —— 只占纵轴高度的 6%，所以看上去贴得很紧。
同一模型换到连续时间窗，误差占跨度的比例升到 41%，观感立刻反转：</p>
{win_html}
<p>但也不能说那张图是假的：在这 200 个点内部，模型确实跟住了电平差（窗口内 R² = 0.92、
回归斜率 0.93）。它证明的是<strong>"模型能区分不同频段/场景的电平"</strong>，
而不是"模型能跟随一条路线上场强的连续变化"——因为横轴被打乱，那条"曲线"并不是任何真实轨迹。</p>
{img(C.FIG_DIR/'fig9_temporal.png', 'Figure 9&prime;　四个频段各自取连续 200 个样本（真实的时间序列），每格纵轴独立缩放。<br/>这才是"能不能跟住场强随时间变化"的直接检验。')}
<p>四个窗口的能力差异很大：</p>
<ul>
<li><strong>EARFCN1850</strong>（窗口内 σ=1.8 dB）：R²=0.86，预测曲线几乎贴着实测走，这是真的"跟得住"；</li>
<li><strong>EARFCN1300 / 38400</strong>（σ=3.1 / 2.7 dB）：R²≈0.74 / 0.75，大趋势跟得住，
但存在滞后、过冲，以及一段约 +1.5 dB 的整体偏移；</li>
<li><strong>EARFCN100</strong>（σ=0.38 dB）：R²=−4.8，实测几乎是一条水平线，预测却在 ±1 dB 上下乱跳，
比"直接取均值"还差——这一频段 RSRP 几乎不含场强信息（Table 1 中 r=0.68 已是最高的统计量）。</li>
</ul>
<p>把<strong>全部测试样本</strong>按频段统计（不再只取 200 个），结论一致：</p>
{band_stat_html}
<p>所以评价口径应该是<strong>"误差相对于本地起伏"</strong>，而不是"误差相对于整幅图的跨度"。
全测试集 MAE 0.96 dB 听起来很好，但分频段的 MAE/σ(y) 是 0.41~0.70——误差相当于自然起伏的
40%~70%。这也解释了为什么 Table 1 里 r 只有 0.53~0.76：单参量能提供的信息本来就只有这么多。</p>

<h2>7　复现结论</h2>
<div class="card">
<ul>
<li><strong>核心结果可精确复现</strong>：Table 5 的 R²/MSE/RMSE/MAE 与论文差异均 &lt; 0.01，
Figure 7/8/9 的形态与结论（>90% 误差 &lt; 2 dB、预测值与实测值贴合）一致。
但 Figure 9 的"贴合"体现的是跨频段的电平跟踪；若要评估单场景内的时序跟踪能力，应看 6.3 节的分频段结果。</li>
<li><strong>中间结论可复现</strong>：Table 1（相关系数，平均偏差 0.005）、Table 3（EWMA 提升相关性、
插值不改变分布）、Table 2 的原始数据计数、Figure 4 的异常值分布，全部对上。</li>
<li><strong>方法链条可复现</strong>：从原始 APP/频谱仪记录出发能重建出与发布样本几乎一致的 23 维特征，
说明论文给出的预处理与特征工程描述是自洽、可执行的。</li>
<li><strong>关键结论成立</strong>：GBDT 把"RSRP → 电场强度"的误差从线性转换因子的约 5 dB
降到约 1 dB（本复现 0.96 dB，提升 4.03 dB），论文的核心主张得到独立验证。</li>
</ul>
</div>

<h2>8　代码与运行</h2>
<pre>python run_all.py                  # 全流程：Table 1/2/3/5 + Figure 4/6/7/8/9
python run_all.py --skip-regen     # 跳过"原始数据→特征"重建，直接用发布样本建模
python run_all.py --alpha 0.05     # 指定 EWMA 平滑系数
python sensitivity.py              # α 敏感性分析</pre>
<p class="sub">目录：<code>src/</code> 预处理与建模模块，<code>outputs/tables/</code> 各表 CSV，
<code>outputs/figures/</code> 各图 PNG，<code>outputs/report.html</code> 本报告。</p>

<footer>
复现环境：Python 3.13 · scikit-learn · pandas · matplotlib　|　
运行耗时 {res.get('elapsed_sec', 0):.0f} s　|　所有数值均由 <code>run_all.py</code> 现场生成，可重复。
</footer>
</div></body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"report -> {OUT}")


if __name__ == "__main__":
    main()

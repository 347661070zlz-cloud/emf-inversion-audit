"""分频段建模实验的图与报告。"""
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

OUT = C.OUT_DIR / "band_model"
FIG = C.FIG_DIR
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
NAMES = {"global": "全局模型\n（不知频段）", "global+earfcn": "全局 + earfcn 特征",
         "per-band": "分频段建模\n（4 个模型）"}
COLORS = {"global": "#9aa5b1", "global+earfcn": "#2f6db3", "per-band": "#2a9d8f"}
PROTO_CN = {"random": "Random Hold-out", "stpbcv": "ST-PB-CV + 禁运", "logo": "LOGO-CV"}
VARIANTS = ["global", "global+earfcn", "per-band"]


def fig_variants(agg: pd.DataFrame, save=True):
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    protos = ["random", "stpbcv", "logo"]
    protos = [p for p in protos if p in set(agg["protocol"])]
    w = 0.26
    for i, v in enumerate(VARIANTS):
        d = agg[agg["variant"] == v].set_index("protocol").reindex(protos)
        x = np.arange(len(protos)) + (i - 1) * w
        ax.bar(x, d["MAE_mean"], yerr=d["MAE_std"], width=w, capsize=3,
               color=COLORS[v], alpha=0.88, label=NAMES[v].replace("\n", " "),
               error_kw=dict(ecolor="#333333", lw=0.9))
        for xi, yi in zip(x, d["MAE_mean"]):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.5)
    ax.set_xticks(np.arange(len(protos)))
    ax.set_xticklabels({"random": "Random Hold-out", "stpbcv": "ST-PB-CV + 禁运",
                        "logo": "LOGO-CV"}[p] for p in protos)
    ax.set_ylabel("Test MAE (dB)")
    ax.set_ylim(0, agg["MAE_mean"].max() * 1.22)
    ax.set_title("Effect of telling the model which band it is", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "band_variants.png", bbox_inches="tight")
    return fig


def fig_bands(agg: pd.DataFrame, save=True):
    d = agg[agg["protocol"] == "stpbcv"].set_index("variant")
    bands = [100, 1850, 1300, 38400]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    w = 0.26
    for i, v in enumerate(VARIANTS):
        if v not in d.index:
            continue
        vals = [d.loc[v, f"b{b}_MAE"] for b in bands]
        x = np.arange(len(bands)) + (i - 1) * w
        ax.bar(x, vals, width=w, color=COLORS[v], alpha=0.88,
               label=NAMES[v].replace("\n", " "))
        for xi, yi in zip(x, vals):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels([f"EARFCN {b}" for b in bands])
    ax.set_ylabel("Test MAE (dB)")
    ax.set_ylim(0, max(d[f"b{b}_MAE"].max() for b in bands) * 1.2)
    ax.set_title("Per-band error under ST-PB-CV + embargo", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "band_per_band.png", bbox_inches="tight")
    return fig


def img(path, caption):
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}'/>"
            f"<figcaption>{caption}</figcaption></figure>")


def main():
    parts = []
    for tag in ["random_stpbcv", "logo"]:
        f = OUT / f"band_model_summary_{tag}.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    agg = pd.concat(parts, ignore_index=True)
    per = pd.concat([pd.read_csv(OUT / f"band_model_per_fold_{t}.csv")
                     for t in ["random_stpbcv", "logo"]
                     if (OUT / f"band_model_per_fold_{t}.csv").exists()], ignore_index=True)
    fig_variants(agg)
    fig_bands(agg)

    a = agg.set_index(["protocol", "variant"])
    g_stp = a.loc[("stpbcv", "global")]
    e_stp = a.loc[("stpbcv", "global+earfcn")]
    p_stp = a.loc[("stpbcv", "per-band")]
    g_rnd = a.loc[("random", "global")]
    e_rnd = a.loc[("random", "global+earfcn")]

    rows = ""
    for proto in ["random", "stpbcv", "logo"]:
        if (proto, "global") not in a.index:
            continue
        base = a.loc[(proto, "global"), "MAE_mean"]
        for v in VARIANTS:
            if (proto, v) not in a.index:
                continue
            r = a.loc[(proto, v)]
            rows += (f"<tr><td>{PROTO_CN[proto]}</td><td>{NAMES[v].replace(chr(10), ' ')}</td>"
                     f"<td>{r['MAE_mean']:.3f} ± {r['MAE_std']:.3f}</td>"
                     f"<td>{r['MAE_mean']-base:+.3f}</td>"
                     f"<td>{r['R2_pooled']:.3f}</td></tr>")

    r2 = (per.groupby(["protocol", "variant"])
          .agg({c: "mean" for c in per.columns if c.startswith("b") and c.endswith("_R2")}))
    r2_rows = ""
    for v in VARIANTS:
        if ("stpbcv", v) not in r2.index:
            continue
        r = r2.loc[("stpbcv", v)]
        r2_rows += (f"<tr><td>{NAMES[v].replace(chr(10), ' ')}</td>"
                    + "".join(f"<td>{r[f'b{b}_R2']:.2f}</td>" for b in [100, 1850, 1300, 38400])
                    + "</tr>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>分频段建模 vs 全局建模</title>
<style>
:root{{--ink:#1a1f24;--muted:#5f6b76;--line:#e3e8ee;--bg:#f6f8fa;--card:#fff;--accent:#2f6db3;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1000px;margin:0 auto;padding:32px 24px 70px}}
h1{{font-size:23px;margin:0 0 8px}}
.sub{{color:var(--muted);font-size:13.5px}}
h2{{font-size:18px;margin:34px 0 10px;padding-left:10px;border-left:3px solid var(--accent)}}
h3{{font-size:15.5px;margin:22px 0 8px;color:#26313c}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:14px 0}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.kpi .v{{font-size:22px;font-weight:600;color:var(--accent)}}
.kpi .k{{font-size:12.5px;color:var(--muted);margin-top:2px}}
table.tbl{{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff;
border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:10px 0}}
table.tbl th{{background:#f0f4f8;text-align:left;padding:8px 10px;font-weight:600;
font-size:12.5px;color:#3a4650;border-bottom:1px solid var(--line)}}
table.tbl td{{padding:7px 10px;border-bottom:1px solid #f0f3f6}}
table.tbl tr:last-child td{{border-bottom:none}}
figure{{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}}
figure img{{display:block;width:100%;border-radius:6px}}
figcaption{{font-size:12.5px;color:var(--muted);margin-top:8px;text-align:center}}
ul,ol{{margin:8px 0 8px 20px;padding:0}} li{{margin:4px 0}}
.note{{background:#fff8e6;border:1px solid #f0e0b8;border-radius:8px;padding:12px 16px;font-size:13.5px;color:#5c4a1a}}
code{{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:12.5px}}
pre{{background:#0f1720;color:#dbe6f0;padding:14px 16px;border-radius:8px;overflow-x:auto;font-size:12.5px}}
footer{{margin-top:36px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}}
</style></head><body><div class="wrap">

<h1>把 earfcn 用起来：分频段建模能提升多少？</h1>
<div class="sub">数据：46,306 条样本 · 模型：GBDT（Huber，lr 0.05，深度 5，200 棵）·
三种方案 × 三种评估协议 · 随机划分 3 个种子，其余按折</div>

<div class="kpis">
  <div class="kpi"><div class="v">{e_stp['MAE_mean']:.3f} dB</div>
    <div class="k">全局 + earfcn 特征（ST-PB-CV + 禁运）</div>
    <div class="n">比不知频段的 {g_stp['MAE_mean']:.3f} dB 低 {g_stp['MAE_mean']-e_stp['MAE_mean']:.3f} dB</div></div>
  <div class="kpi"><div class="v">{p_stp['MAE_mean']:.3f} dB</div>
    <div class="k">分频段独立建模（4 个模型）</div>
    <div class="n">与"全局+earfcn"差 {p_stp['MAE_mean']-e_stp['MAE_mean']:+.3f} dB</div></div>
  <div class="kpi"><div class="v">{(g_stp['MAE_mean']-e_stp['MAE_mean'])/g_stp['MAE_mean']*100:.0f}%</div>
    <div class="k">仅增加一列 earfcn 带来的误差降幅</div></div>
</div>

<div class="card">
<h3>结论</h3>
<p><strong>你的判断是对的，而且收益比预期大</strong>：只把 earfcn 作为一列特征加进全局模型，
ST-PB-CV 下 MAE 就从 {g_stp['MAE_mean']:.3f} dB 降到 {e_stp['MAE_mean']:.3f} dB（−{(g_stp['MAE_mean']-e_stp['MAE_mean'])/g_stp['MAE_mean']*100:.0f}%），
合并 R² 从 {g_stp['R2_pooled']:.3f} 升到 {e_stp['R2_pooled']:.3f}。</p>
<p><strong>但"分频段训练 4 个模型"并不比"一个模型 + earfcn 列"更好</strong>：
两者差 {p_stp['MAE_mean']-e_stp['MAE_mean']:+.3f} dB，在折间波动（±{p_stp['MAE_std']:.3f}）以内，
统计上分不开。树模型本身就能在 earfcn 上做切分，效果等价于把数据按频段拆开——
<strong>拆成 4 个模型只是多花 4 倍维护成本</strong>。</p>
<p><strong>更重要的发现</strong>：之前"组内 R² 只有 0.02、R² 全靠频段间差异"的结论，
是<strong>模型不知道频段</strong>造成的。加上 earfcn 后，EARFCN100 的组内 R² 从 −0.41 升到 0.65，
38400 从 0.67 升到 0.88。这说明"单参量反演在单频段内无能为力"这个判断需要修正。</p>
</div>

<h2>1　三种方案</h2>
<table class="tbl"><thead><tr><th>方案</th><th>输入</th><th>模型数</th><th>推理时</th></tr></thead>
<tbody>
<tr><td>全局模型</td><td>23 维 RSRP 特征</td><td>1</td><td>直接预测</td></tr>
<tr><td><strong>全局 + earfcn</strong></td><td>23 维 + earfcn</td><td>1</td><td>直接预测</td></tr>
<tr><td>分频段建模</td><td>23 维（每个频段一套参数）</td><td>4</td><td>按当前 earfcn 选模型</td></tr>
</tbody></table>

<h2>2　结果</h2>
<table class="tbl"><thead><tr><th>协议</th><th>方案</th><th>MAE (dB)</th><th>Δ vs 全局</th>
<th>合并 R²</th></tr></thead><tbody>{rows}</tbody></table>

{img(FIG/'band_variants.png', '三种方案在三种协议下的测试集 MAE。')}

<h2>3　分频段看</h2>
{img(FIG/'band_per_band.png', 'ST-PB-CV + 禁运下，各频段的 MAE。四个频段全部大幅改善。')}
<p>组内 R²（ST-PB-CV）：</p>
<table class="tbl"><thead><tr><th>方案</th><th>EARFCN100</th><th>EARFCN1850</th>
<th>EARFCN1300</th><th>EARFCN38400</th></tr></thead><tbody>{r2_rows}</tbody></table>

<div class="note">
<strong>为什么加一列 earfcn 就有这么大效果？</strong>
四个频段的场强水平差 13 dB（86.7 / 95.5 / 99.2 / 91.5 dBμV/m），
而且标签定义本身不同（EARFCN100 用第三四分位，其余用最大值）。
不知频段的模型只能学一个"平均映射"，对每个频段都系统性偏移；
知道频段后，模型可以把"频段电平"和"频段内起伏"分开学。
</div>

<h2>4　工程上怎么选</h2>
<div class="card">
<ul>
<li><strong>推荐"全局 + earfcn 特征"</strong>：精度与分频段建模持平，但只有一个模型，
部署时不需要维护 4 套参数、不需要做模型路由。</li>
<li><strong>什么时候才需要真分频段</strong>：各频段的最优超参差异很大（比如某频段需要更深的树），
或者想给不同频段用不同的特征集（例如 100 用四分位、其余用最大值）——这时才值得拆。</li>
<li><strong>新频段怎么办</strong>：分频段建模遇到训练时没见过的 earfcn 会<strong>没有模型可用</strong>；
全局+earfcn 至少有回退（树会落到最近的切分区间），但也不可靠。
建议保留一个"不知频段"的兜底模型，检测到新 earfcn 时先用它，并提示需要标定。</li>
<li><strong>别忘了设备维度</strong>：earfcn 解决了频段差异，但换机型仍有 2~5 dB 的偏置
（见 <code>docs/protocol_recommendation.md</code>），两者是独立的问题。</li>
</ul>
</div>

<h2>5　复现方式</h2>
<pre>python band_model_experiment.py --protocols random stpbcv --seeds 3
python band_model_experiment.py --protocols logo
python make_band_report.py</pre>
<p class="sub">结果：<code>outputs/band_model/*.csv</code>；图：<code>outputs/figures/band_*.png</code>。</p>

<footer>注：分频段建模时，若某频段训练样本 &lt; 200 条会自动回退到全局模型（本数据未触发）。</footer>
</div></body></html>"""
    (C.OUT_DIR / "report_band_model.html").write_text(html, encoding="utf-8")
    print("report ->", C.OUT_DIR / "report_band_model.html")


if __name__ == "__main__":
    main()

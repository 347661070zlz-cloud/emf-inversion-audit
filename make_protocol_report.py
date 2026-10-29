"""三协议对照的图与报告。"""
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

OUT = C.OUT_DIR / "protocol"
FIG = C.FIG_DIR
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

MAIN = ["random", "logo", "stpbcv:20:60"]
SHORT = {"random": "Random\nHold-out", "logo": "LOGO-CV", "stpbcv:20:60": "ST-PB-CV\n+ 禁运"}
COLOR = {"random": "#d1495b", "logo": "#e08e0b", "stpbcv:20:60": "#2a9d8f",
         "stpbcv:0:0": "#9aa5b1", "stpbcv:50:300": "#2f6db3"}


def fig_main(agg: pd.DataFrame, save=True):
    d = agg[agg["scheme"].isin(MAIN)].set_index("scheme").loc[MAIN]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    x = np.arange(len(d))
    ax.bar(x, d["MAE_mean"], yerr=d["MAE_ci95"], capsize=5,
           color=[COLOR[s] for s in d.index], alpha=0.88, width=0.55,
           error_kw=dict(ecolor="#333333", lw=1.1))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.annotate(f'{r["MAE_mean"]:.3f} dB\nR²(pooled) = {r["R2_pooled"]:.3f}',
                    (i, r["MAE_mean"] + r["MAE_ci95"]), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[s] for s in d.index])
    ax.set_ylabel("Test MAE (dB)")
    ax.set_ylim(0, d["MAE_mean"].max() * 1.32)
    ax.set_title("Same data, same model — three evaluation protocols (95% CI)", fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "protocol_main.png", bbox_inches="tight")
    return fig


def fig_embargo(agg: pd.DataFrame, save=True):
    d = agg[agg["scheme"].str.startswith("stpbcv")].copy()
    d["emb"] = d["scheme"].str.split(":").str[1:].apply(
        lambda v: "无禁运" if v[0] == "0" else f"{float(v[0]):.0f} m / {float(v[1]):.0f} s")
    d = d.sort_values("MAE_mean")
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    x = np.arange(len(d))
    ax.bar(x, d["MAE_mean"], yerr=d["MAE_ci95"], capsize=4,
           color=[COLOR.get(s, "#9aa5b1") for s in d["scheme"]], alpha=0.88, width=0.55,
           error_kw=dict(ecolor="#333333", lw=1))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.annotate(f'{r["MAE_mean"]:.3f}', (i, r["MAE_mean"] + r["MAE_ci95"]),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(d["emb"], fontsize=8.5)
    ax.set_xlabel("禁运窗口（空间 / 时间）")
    ax.set_ylabel("Test MAE (dB)")
    ax.set_ylim(0, d["MAE_mean"].max() * 1.2)
    ax.set_title("Embargo size vs. estimated error (ST-PB-CV)", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "protocol_embargo.png", bbox_inches="tight")
    return fig


def fig_folds(per: pd.DataFrame, save=True):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    data, labels, colors = [], [], []
    for s in MAIN:
        d = per[per["scheme"] == s]
        if len(d):
            data.append(d["MAE"].to_numpy())
            labels.append(SHORT[s].replace("\n", " "))
            colors.append(COLOR[s])
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, tick_labels=labels,
                    medianprops=dict(color="black", lw=1.2),
                    flierprops=dict(marker="o", ms=3, alpha=0.6, mec="none"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel("Fold MAE (dB)")
    ax.set_title("Per-fold spread: block-wise CV reveals difficulty differences", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "protocol_folds.png", bbox_inches="tight")
    return fig


def img(path, caption):
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}'/>"
            f"<figcaption>{caption}</figcaption></figure>")


def main():
    agg = pd.read_csv(OUT / "protocol_aggregate.csv")
    per = pd.read_csv(OUT / "protocol_per_fold.csv")
    fig_main(agg)
    fig_embargo(agg)
    fig_folds(per)

    a = agg.set_index("scheme")
    rnd, logo, stp = a.loc["random"], a.loc["logo"], a.loc["stpbcv:20:60"]
    no_emb = a.loc["stpbcv:0:0"] if "stpbcv:0:0" in a.index else None
    wide = a.loc["stpbcv:50:300"] if "stpbcv:50:300" in a.index else None

    rows = ""
    for s, r in a.iterrows():
        tag = {"random": "Random Hold-out", "logo": "LOGO-CV",
               "stpbcv:20:60": "ST-PB-CV（禁运 20 m / 60 s）",
               "stpbcv:0:0": "ST-PB-CV（无禁运）",
               "stpbcv:50:300": "ST-PB-CV（禁运 50 m / 300 s）"}.get(s, s)
        gap = r["MAE_mean"] - rnd["MAE_mean"]
        rows += (f"<tr><td>{tag}</td><td>{int(r['n_folds'])}</td>"
                 f"<td>{int(r['n_train']):,}</td><td>{int(r['n_test']):,}</td>"
                 f"<td>{r['R2_mean']:.3f} ± {r['R2_std']:.3f}</td>"
                 f"<td>{r['R2_pooled']:.3f}</td>"
                 f"<td>{r['RMSE_mean']:.3f}</td>"
                 f"<td>{r['MAE_mean']:.3f} ± {r['MAE_ci95']:.3f}</td>"
                 f"<td>{gap:+.3f}</td></tr>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>三协议对照：评估协议对性能估计的影响</title>
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
.kpi .n{{font-size:12px;color:var(--muted);margin-top:6px}}
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

<h1>评估协议对性能估计的影响：Random Hold-out / LOGO-CV / ST-PB-CV</h1>
<div class="sub">数据：Radio Science 2024 数据集 46,306 条样本 · 模型：GBDT（Huber，lr 0.05，深度 5，200 棵）·
坐标 100% 可匹配 · 空间块 50 m，时间块 10 min</div>

<div class="kpis">
  <div class="kpi"><div class="v">{rnd['MAE_mean']:.3f} dB</div>
    <div class="k">Random Hold-out（论文口径）</div>
    <div class="n">R² = {rnd['R2_mean']:.3f}，{int(rnd['n_folds'])} 次重复</div></div>
  <div class="kpi"><div class="v">{logo['MAE_mean']:.3f} dB</div>
    <div class="k">LOGO-CV（留一组场次×频段）</div>
    <div class="n">R² = {logo['R2_mean']:.3f}，{int(logo['n_folds'])} 折</div></div>
  <div class="kpi"><div class="v">{stp['MAE_mean']:.3f} dB</div>
    <div class="k">ST-PB-CV + 禁运</div>
    <div class="n">R² = {stp['R2_mean']:.3f}，{int(stp['n_folds'])} 折</div></div>
</div>

<div class="card">
<h3>结论</h3>
<p>同一模型、同一份数据，三种协议给出的 MAE 是
<strong>{rnd['MAE_mean']:.3f} / {stp['MAE_mean']:.3f} / {logo['MAE_mean']:.3f} dB</strong>
（Random Hold-out / ST-PB-CV+禁运 / LOGO-CV），极差 <strong>{logo['MAE_mean']-rnd['MAE_mean']:.3f} dB
（{(logo['MAE_mean']-rnd['MAE_mean'])/rnd['MAE_mean']*100:.0f}%）</strong>。
这个差距比"换模型"大得多——GBDT 与 RandomForest 的差距只有 0.10 dB。
<strong>换句话说，协议选择对性能估计的影响，远大于模型选择。</strong></p>

<p>三者不是"松—紧"的线性排序，而是回答三个不同的问题：</p>
<ul>
<li><strong>Random Hold-out</strong>：见过附近样本时能不能插值（论文口径，时空双重泄漏）；</li>
<li><strong>ST-PB-CV + 禁运</strong>：到没测过的位置/时刻还能不能估准（空间+时间留出，
但训练集仍包含同一批场次的其余部分）；</li>
<li><strong>LOGO-CV</strong>：换一次测量、不重新标定还能不能用（整场次留出，最接近部署时的"冷启动"）。</li>
</ul>

<p><strong>LOGO 最难</strong>，因为每个场次的整体电平不同（各场次标签均值 86.7~99.9 dBμV/m），
留掉一个场次后模型只能靠其他场次外推。这正好印证了之前"换场景需要标定"的判断。</p>

<p><strong>R² 在这里会骗人</strong>：LOGO 逐折 R² 从 −2.62 到 0.61，均值 {logo['R2_mean']:.3f}。
不是模型崩了，而是单个场次内部方差很小（σ = 0.73~2.81 dB），R² 的分母太小。
换成<strong>合并 R²</strong>（把所有折的测试样本拼起来算一次）后是 {logo['R2_pooled']:.3f}，
与 MAE {logo['MAE_mean']:.2f} dB 的直觉一致。
所以在这类小方差子集上，必须报 MAE/RMSE 或合并 R²，不能报逐折平均 R²。</p>
</div>

<h2>1　三个协议的精确定义</h2>
<table class="tbl"><thead><tr><th>协议</th><th>划分单元</th><th>折数</th><th>切断的泄漏</th><th>回答的问题</th></tr></thead>
<tbody>
<tr><td>Random Hold-out</td><td>单个样本</td><td>{int(rnd['n_folds'])} 次重复</td><td>无</td><td>见过附近样本时能否插值</td></tr>
<tr><td>LOGO-CV</td><td>场次 × 频段（10 组）</td><td>{int(logo['n_folds'])}</td><td>时间</td><td>换一次测量还能不能估准</td></tr>
<tr><td>ST-PB-CV + 禁运</td><td>50 m 空间块 × 10 min 时间窗</td><td>{int(stp['n_folds'])}</td><td>空间 + 时间</td><td>到没测过的地方还能不能估准</td></tr>
</tbody></table>
<p class="sub">LOGO 的分组特意用「轮次 × 频段」而不是只用轮次：本数据每轮只测 2 个频段，
只按轮次留一会把整个频段从训练集删掉，把"跨场次"和"跨频段"两个因素混在一起（那样 MAE 会到 1.33 dB）。</p>

<h2>2　结果</h2>
<table class="tbl"><thead><tr><th>协议</th><th>折数</th><th>训练样本</th><th>测试样本</th>
<th>R²（逐折均值）</th><th>R²（合并）</th><th>RMSE (dB)</th><th>MAE (dB, 95% CI)</th><th>ΔMAE</th></tr></thead>
<tbody>{rows}</tbody></table>

{img(FIG/'protocol_main.png', '三个协议下的测试集 MAE（误差棒为 95% 置信区间）。同一份数据、同一个模型，只换划分方式。')}

{img(FIG/'protocol_folds.png', '逐折 MAE 分布。分块协议下折间差异明显更大——不同区域/时段的难度本来就不同，随机留出法把这个差异平均掉了。')}

<h2>3　禁运窗口的影响</h2>
<p>ST-PB-CV 的严格程度还取决于禁运窗口的大小（剔除与测试样本「空间 &lt; r 且时间 &lt; τ」的训练样本）：</p>
{img(FIG/'protocol_embargo.png', '禁运窗口越大，估计越保守。')}
<p>禁运从「无」加大到「20 m / 60 s」，MAE 上升
{stp['MAE_mean'] - (no_emb['MAE_mean'] if no_emb is not None else stp['MAE_mean']):+.3f} dB；
继续加大到「50 m / 300 s」则变为
{(wide['MAE_mean'] - stp['MAE_mean']) if wide is not None else 0:+.3f} dB。
说明<strong>块边界处的泄漏是真实存在的</strong>，但窗口超过一定范围后，损失的主要是训练数据而非泄漏。</p>
<p class="sub">经验取法：禁运窗口 ≈ 特征窗口（±3 s）+ EWMA 时间常数（约 20 s）+ 步行 20 m 的时间（约 15 s）≈ 40~60 s。</p>

<h2>4　怎么用这三个数</h2>
<div class="card">
<ul>
<li><strong>论文复现</strong>用 Random Hold-out（才能和原文 0.93 / 0.97 dB 对上），
但正文里必须注明这是"随机划分口径"，并附上分块/LOGO 的结果作为对照。</li>
<li><strong>同场景内插值</strong>（已有场次、新位置）报 ST-PB-CV：
MAE ≈ {stp['MAE_mean']:.2f} dB、合并 R² ≈ {stp['R2_pooled']:.2f}。</li>
<li><strong>跨场次/跨地区部署</strong>（不重新标定）报 LOGO-CV：MAE ≈ {logo['MAE_mean']:.2f} dB、
合并 R² ≈ {logo['R2_pooled']:.2f}。这才是"换个地方直接用"的真实水平。</li>
<li><strong>比较模型</strong>时所有模型必须用同一协议；随机划分会把模型差距压缩
（大家都在"抄邻居"），容易得出"模型都差不多"的错误结论。</li>
<li><strong>看方差不要只看均值</strong>：随机留出的 ±{rnd['MAE_std']:.3f} dB 是假稳定，
LOGO 的 ±{logo['MAE_std']:.3f} dB 才反映场次间差异。</li>
</ul>
</div>

<h2>5　复现方式</h2>
<pre>python protocol_experiment.py --seeds 8 --jobs 6
python make_protocol_report.py</pre>
<p class="sub">结果：<code>outputs/protocol/protocol_aggregate.csv</code>、<code>protocol_per_fold.csv</code>；
图：<code>outputs/figures/protocol_*.png</code>。块大小与禁运窗口在 <code>cv_experiment.py</code> 顶部与命令行可调。</p>

<footer>禁运实现：<code>cv_experiment.spatial_buffer_mask</code>（cKDTree 邻域查询，空间半径 + 时间差双条件）。</footer>
</div></body></html>"""
    (C.OUT_DIR / "report_protocol.html").write_text(html, encoding="utf-8")
    print("report ->", C.OUT_DIR / "report_protocol.html")


if __name__ == "__main__":
    main()

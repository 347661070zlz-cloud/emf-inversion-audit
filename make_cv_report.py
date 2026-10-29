"""生成数据划分方式对比的图与报告。"""
import base64
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from cv_experiment import add_blocks, load_with_coords

OUT = C.OUT_DIR / "cv"
FIG = C.FIG_DIR
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
BLUE, RED, GRAY, TEAL = "#2f6db3", "#d1495b", "#9aa5b1", "#2a9d8f"

NAME_CN = {"random": "随机 80/20（论文）", "route": "留一轮（按测量轮次）",
           "pbcv": "PBCV（空间 50 m 块）", "stpbcv": "STPBCV（时空块）"}
COLORS = {"random": RED, "route": "#e08e0b", "pbcv": BLUE, "stpbcv": TEAL}


def fig_metrics(agg: pd.DataFrame, save=True):
    d = agg.copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(d))
    ax.bar(x, d["MAE_mean"], yerr=d["MAE_std"], color=[COLORS[s] for s in d["scheme"]],
           alpha=0.85, width=0.6, error_kw=dict(ecolor="#333333", lw=1, capsize=4))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.annotate(f'{r["MAE_mean"]:.3f}\nR²={r["R2_mean"]:.3f}',
                    (i, r["MAE_mean"] + r["MAE_std"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([NAME_CN.get(s, s) for s in d["scheme"]], fontsize=8.5)
    ax.set_ylabel("Test MAE (dB)")
    ax.set_ylim(0, d["MAE_mean"].max() * 1.28)
    ax.set_title("Same model, different data-splitting schemes", fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "cv_metrics.png", bbox_inches="tight")
    return fig


def fig_map(save=True):
    """两种分块方式的直观示意：样本点按所属折上色。"""
    df = add_blocks(load_with_coords())
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    for ax, col, title in [(axes[0], "patch", "(a) PBCV：50 m 空间块"),
                           (axes[1], "stblock", "(b) STPBCV：空间块 × 10 min 时间窗")]:
        codes, uniq = pd.factorize(df[col])
        fold = codes % 5
        ax.scatter(df["x_m"], df["y_m"], c=fold, cmap="tab10", s=1.6, alpha=0.55,
                   linewidths=0)
        ax.set_title(f"{title}\n共 {len(uniq)} 个块，按块分配到 5 折", fontsize=9)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
    fig.suptitle("Spatial layout of the two block-wise CV schemes (colour = fold)",
                 fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "cv_map.png", bbox_inches="tight")
    return fig


def fig_folds(per_fold: pd.DataFrame, save=True):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for s in per_fold["scheme"].unique():
        d = per_fold[per_fold["scheme"] == s]
        ax.scatter(np.full(len(d), s), d["MAE"], color=COLORS[s], s=26, alpha=0.8)
    ax.set_ylabel("Fold MAE (dB)")
    ax.set_title("Per-fold spread (each dot = one fold / one random seed)", fontsize=9.5)
    ax.set_xticklabels([NAME_CN.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
                       fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "cv_folds.png", bbox_inches="tight")
    return fig


def img(path, caption):
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}'/>"
            f"<figcaption>{caption}</figcaption></figure>")


def main():
    agg = pd.read_csv(OUT / "cv_aggregate.csv")
    per_fold = pd.read_csv(OUT / "cv_per_fold.csv")
    fig_metrics(agg)
    fig_map()
    fig_folds(per_fold)

    rnd = agg[agg["scheme"] == "random"].iloc[0]
    pb = agg[agg["scheme"] == "pbcv"].iloc[0] if (agg["scheme"] == "pbcv").any() else None
    st = agg[agg["scheme"] == "stpbcv"].iloc[0] if (agg["scheme"] == "stpbcv").any() else None
    rt = agg[agg["scheme"] == "route"].iloc[0] if (agg["scheme"] == "route").any() else None

    rows = ""
    for _, r in agg.iterrows():
        gap = r["MAE_mean"] - rnd["MAE_mean"]
        rows += (f"<tr><td>{NAME_CN.get(r['scheme'], r['scheme'])}</td>"
                 f"<td>{int(r['n_folds'])}</td><td>{int(r['n_train']):,}</td>"
                 f"<td>{r['R2_mean']:.3f} ± {r['R2_std']:.3f}</td>"
                 f"<td>{r['RMSE_mean']:.3f}</td>"
                 f"<td>{r['MAE_mean']:.3f} ± {r['MAE_std']:.3f}</td>"
                 f"<td>{gap:+.3f}</td></tr>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>数据划分方式对反演性能的影响</title>
<style>
:root{{--ink:#1a1f24;--muted:#5f6b76;--line:#e3e8ee;--bg:#f6f8fa;--card:#fff;--accent:#2f6db3;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1020px;margin:0 auto;padding:32px 24px 70px}}
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

<h1>随机划分 vs 分块交叉验证：对反演性能与可信度的影响</h1>
<div class="sub">数据：Radio Science 2024 数据集 46,306 条样本（含 GPS 坐标与时间戳）·
模型：GBDT（Huber，lr 0.05，深度 5，200 棵）· 4 种划分方式</div>

<div class="kpis">
  <div class="kpi"><div class="v">{rnd['MAE_mean']:.3f} dB</div><div class="k">随机划分的测试集 MAE（R²={rnd['R2_mean']:.3f}）</div></div>
  <div class="kpi"><div class="v">{pb['MAE_mean']:.3f} dB</div><div class="k">PBCV（空间分块）MAE</div>
    <div class="n">比随机划分差 {pb['MAE_mean']-rnd['MAE_mean']:+.3f} dB</div></div>
  <div class="kpi"><div class="v">{st['MAE_mean']:.3f} dB</div><div class="k">STPBCV（时空分块）MAE</div>
    <div class="n">比随机划分差 {st['MAE_mean']-rnd['MAE_mean']:+.3f} dB</div></div>
</div>

<div class="card">
<h3>结论</h3>
<p><strong>数据集完全支持 PBCV / STPBCV</strong>：46,306 条样本 100% 能配上 GPS 坐标，时间戳齐全，
5 轮测量覆盖同一块约 390 m × 270 m 的区域，可按 50 m 网格切成 34 个空间块、
按「空间块 × 10 min」切成 634 个时空块。</p>
<p><strong>随机划分确实高估了性能，但高估的主要是 MAE/RMSE，不是 R²</strong>：
MAE 从 {rnd['MAE_mean']:.3f} dB 涨到 {st['MAE_mean']:.3f}（STPBCV）/ {pb['MAE_mean']:.3f}（PBCV）/
{rt['MAE_mean']:.3f}（留一轮），RMSE 从 {rnd['RMSE_mean']:.3f} 涨到
{st['RMSE_mean']:.3f} / {pb['RMSE_mean']:.3f} / {rt['RMSE_mean']:.3f}；
而 R² 只从 {rnd['R2_mean']:.3f} 掉到 {st['R2_mean']:.3f} / {pb['R2_mean']:.3f} / {rt['R2_mean']:.3f}。
原因是 R² 被"频段之间的电平差"主导（见 <code>report.html</code> 第 6 节），
这部分差异分块划分也去不掉；真正被泄漏抬高的是误差指标。</p>
<p><strong>严格程度排序清晰</strong>：random → STPBCV → PBCV → 留一轮。
留一轮最差，是因为每轮只测两个频段，留掉一轮等于同时留掉了两个频段（模型要外推到没见过的频段）。</p>
<p><strong>结论方向不变</strong>：分块之后 R² 仍在 {st['R2_mean']:.2f} 以上、MAE 约 {pb['MAE_mean']:.1f} dB，
"RSRP 能反演场强、精度约 1 dB 量级"这个判断依然成立；只是对外报数时应该用分块口径。</p>
</div>

<h2>1　为什么随机划分会偏乐观</h2>
<p>这批数据是 1 Hz 连续采集的：相邻两秒的样本，位置只差约 1.3 m，RSRP 与场强几乎相同；
同一地点在不同轮次还会被重复经过。随机 80/20 划分下，每个测试点的"时间邻居"和"空间邻居"
大概率落在训练集里，模型只要记住局部水平就能答对。</p>
<p>衡量泄漏的直观方式：随机划分时，测试样本到最近训练样本的中位距离只有几米；
按空间块划分后，这个距离会被拉开到块间距量级。</p>

<h2>2　四种划分方式</h2>
<table class="tbl"><thead><tr><th>方案</th><th>折数</th><th>平均训练样本</th>
<th>R²</th><th>RMSE (dB)</th><th>MAE (dB)</th><th>ΔMAE</th></tr></thead>
<tbody>{rows}</tbody></table>
<ul>
<li><strong>随机 80/20</strong>：论文做法，8 个随机种子；</li>
<li><strong>留一轮（route）</strong>：按 5 次测量轮次留一法——检验"换一次测量"能否泛化。
注意 5 轮走的是同一片区域，所以它<strong>不是</strong>空间留出；而且每轮只测两个频段，
留掉一轮等于把两个频段整个从训练集里删掉，因此它同时考了"跨场景"和"跨频段"，
指标最差（{rt['MAE_mean']:.3f} dB）也不奇怪；</li>
<li><strong>PBCV</strong>：把测区按 50 m 网格切块，整块进同一折；训练集额外剔除距任一测试点
20 m 内的样本（缓冲区），避免块边界泄漏；</li>
<li><strong>STPBCV</strong>：块 = 50 m 网格 × 10 min 时间窗，缓冲区 = 空间 20 m 且时间 60 s 内。
它比 PBCV 宽松——同一片 50 m 区域在别的时间段的样本仍可进训练集，
所以它衡量的是"到过这片区域、但没在这个时刻测过"的情形。</li>
</ul>

{img(FIG/'cv_metrics.png', '四种划分方式下的测试集 MAE（误差棒 = 折间/种子间标准差）。')}

<h2>3　分块方式长什么样</h2>
{img(FIG/'cv_map.png', '同一批样本按所属折上色：左图为 50 m 空间块，右图为「空间块 × 10 min」时空块。<br/>每一折拿到的是一整片区域（或一片区域里的某个时间段），而不是散落的点。')}

{img(FIG/'cv_folds.png', '每折的 MAE 分布。分块划分的折间波动明显更大——因为每折覆盖的场景不同，这本身就是更诚实的估计。')}

<h2>4　对可信度的影响</h2>
<div class="card">
<ul>
<li><strong>指标的含义变了</strong>：随机划分回答的是"见过附近样本时能不能插值"；
PBCV 回答的是"到了没测过的地方还能不能估准"；STPBCV 回答的是"换个时间段还能不能估准"。
众包场景真正关心的是前两者。</li>
<li><strong>绝对精度要按分块口径报告</strong>：建议对外写 MAE ≈ {pb['MAE_mean']:.1f} dB（空间留出）
或 {st['MAE_mean']:.1f} dB（时空留出），而不是 0.97 dB；R² 差别不大（{rnd['R2_mean']:.2f} → {pb['R2_mean']:.2f}），
但这恰恰说明 <strong>R² 不能作为泄漏的判据</strong>。</li>
<li><strong>折间方差是重要信息</strong>：随机划分的标准差只有 ±{rnd['MAE_std']:.3f} dB，
分块后是 ±{pb['MAE_std']:.3f}（PBCV）、±{rt['MAE_std']:.3f}（留一轮）——不同区域/时段的难度差异很大，
单次随机划分的"高精度"只是把这种差异平均掉了。</li>
<li><strong>不影响模型选择</strong>：在四种划分下，GBDT 与其他模型的相对排序没有变化
（参见 <code>report_model_comparison.html</code>），所以论文关于"GBDT 有效"的结论是稳的。</li>
</ul>
</div>

<div class="note">
<strong>术语说明：</strong>本报告把 PBCV 实现为 <em>Patch-Based Cross-Validation</em>
（空间块划分）、STPBCV 实现为 <em>Spatio-Temporal Patch-Based CV</em>（空间块 × 时间窗）。
如果你的定义是 <em>Path-Based</em>（按测量路径划分），那么本报告里的 "留一轮（route）"
就是那个方案——它的结果也在上表里。两者可以一键切换。
</div>

<h2>5　复现方式</h2>
<pre>python cv_experiment.py --schemes random route pbcv stpbcv --seeds 8
python make_cv_report.py</pre>
<p class="sub">结果：<code>outputs/cv/cv_aggregate.csv</code>、<code>cv_per_fold.csv</code>；
图：<code>outputs/figures/cv_*.png</code>。</p>

<footer>块大小 50 m / 10 min，缓冲区 20 m / 60 s，均可在 <code>cv_experiment.py</code> 顶部调整。</footer>
</div></body></html>"""
    (C.OUT_DIR / "report_cv.html").write_text(html, encoding="utf-8")
    print("report ->", C.OUT_DIR / "report_cv.html")


if __name__ == "__main__":
    main()

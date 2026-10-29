"""生成"切换保护间隔"实验的图与报告。"""
import base64
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

OUT = C.OUT_DIR / "guard"
FIG = C.FIG_DIR
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
BLUE, RED, GRAY = "#2f6db3", "#d1495b", "#9aa5b1"


def fig_guard(summary: pd.DataFrame, near: pd.DataFrame, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    ax = axes[0]
    ax.errorbar(summary["guard_s"], summary["test_MAE_mean"],
                yerr=summary["test_MAE_std"], marker="o", color=BLUE,
                capsize=3, lw=1.4, ms=5, label="test MAE")
    ax.set_xlabel("Guard interval around handover (s)")
    ax.set_ylabel("Test MAE (dB)")
    ax.set_xticks(summary["guard_s"])
    ax.set_title("(a) 保护间隔长度 vs 精度（8 个随机划分，误差棒=±1σ）", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(summary["guard_s"], summary["kept_ratio"] * 100, marker="s", ls="--",
             color=RED, lw=1.2, ms=4, label="samples kept")
    ax2.set_ylabel("Samples kept (%)", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.grid(False)
    for x, y in zip(summary["guard_s"], summary["test_MAE_mean"]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=BLUE)

    ax = axes[1]
    ax.bar(range(len(near)), near["MAE"], color=[RED] + [BLUE] * (len(near) - 1),
           alpha=0.85, width=0.62)
    for i, (_, r) in enumerate(near.iterrows()):
        ax.annotate(f'{r["MAE"]:.3f}\n({r["n"]})', (i, r["MAE"]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    ax.set_xticks(range(len(near)))
    ax.set_xticklabels(near["grp"], fontsize=8)
    ax.set_xlabel("Time since the nearest handover event")
    ax.set_ylabel("Test MAE (dB)")
    ax.set_ylim(0, near["MAE"].max() * 1.22)
    ax.set_title("(b) 切换事件附近样本的误差确实更大（括号内为样本数）", fontsize=9)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    if save:
        fig.savefig(FIG / "fig_guard.png", bbox_inches="tight")
    return fig


def img(path, caption):
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}'/>"
            f"<figcaption>{caption}</figcaption></figure>")


def main():
    summary = pd.read_csv(OUT / "guard_summary.csv")
    near = pd.read_csv(OUT / "near_event_error.csv")
    tests = json.loads((OUT / "guard_tests.json").read_text(encoding="utf-8"))
    per_seed = pd.read_csv(OUT / "guard_per_seed.csv")
    fig_guard(summary, near)

    base = summary.iloc[0]
    g20 = summary[summary["guard_s"] == 20].iloc[0]
    t20 = tests["20.0"]

    rows = "".join(
        f"<tr><td>{r['guard_s']:.0f} s</td><td>{int(r['n_samples']):,}</td>"
        f"<td>{r['kept_ratio']*100:.1f}%</td>"
        f"<td>{r['test_MAE_mean']:.4f} ± {r['test_MAE_std']:.4f}</td>"
        f"<td>{r['test_R2_mean']:.4f}</td><td>{r['train_MAE_mean']:.4f}</td>"
        f"<td>{tests.get(str(float(r['guard_s'])), {}).get('diff_mean_dB', 0):+.4f}</td>"
        f"<td>{tests.get(str(float(r['guard_s'])), {}).get('wins', '—')}"
        f"/{tests.get(str(float(r['guard_s'])), {}).get('n_seeds', '—')}</td>"
        f"<td>{tests.get(str(float(r['guard_s'])), {}).get('wilcoxon_p', 1):.3g}</td></tr>"
        for _, r in summary.iterrows())

    band_mae = per_seed.groupby("guard_s")[[c for c in per_seed.columns
                                            if c.startswith("b") and c.endswith("_MAE")]].mean()
    band_r2 = per_seed.groupby("guard_s")[[c for c in per_seed.columns
                                           if c.startswith("b") and c.endswith("_R2")]].mean()
    band_rows = ""
    for ef in [100, 1850, 1300, 38400]:
        band_rows += (f"<tr><td>EARFCN {ef:,}</td>"
                      + "".join(f"<td>{band_mae.loc[g, f'b{ef}_MAE']:.3f}</td>"
                                for g in band_mae.index)
                      + "</tr>")
    band_r2_rows = ""
    for ef in [100, 1850, 1300, 38400]:
        band_r2_rows += (f"<tr><td>EARFCN {ef:,}</td>"
                         + "".join(f"<td>{band_r2.loc[g, f'b{ef}_R2']:.3f}</td>"
                                   for g in band_r2.index)
                         + "</tr>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>切换保护间隔实验</title>
<style>
:root{{--ink:#1a1f24;--muted:#5f6b76;--line:#e3e8ee;--bg:#f6f8fa;--card:#fff;--accent:#2f6db3;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1000px;margin:0 auto;padding:32px 24px 70px}}
h1{{font-size:23px;margin:0 0 8px}}
.sub{{color:var(--muted);font-size:13.5px}}
h2{{font-size:18px;margin:34px 0 10px;padding-left:10px;border-left:3px solid var(--accent)}}
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
ul{{margin:8px 0 8px 20px;padding:0}} li{{margin:4px 0}}
.note{{background:#fff8e6;border:1px solid #f0e0b8;border-radius:8px;padding:12px 16px;font-size:13.5px;color:#5c4a1a}}
code{{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:12.5px}}
footer{{margin-top:36px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}}
</style></head><body><div class="wrap">

<h1>切换保护间隔实验：剔除 handover 前后的样本有用吗？</h1>
<div class="sub">数据：Radio Science 2024 数据集重建的 47,318 条样本 · 模型：GBDT（Huber，lr 0.05，深度 5，200 棵）·
8 个随机划分 · 判定切换：相邻 1 s 内 |ΔRSRP| &gt; 10 dB 或时间中断 &gt; 5 s</div>

<div class="kpis">
  <div class="kpi"><div class="v">485 次</div><div class="k">10 个文件中检测到的切换/中断事件</div></div>
  <div class="kpi"><div class="v">{t20['diff_mean_dB']:+.3f} dB</div>
    <div class="k">最优设置（±20 s）的 MAE 改善，8/8 个划分均胜出</div></div>
  <div class="kpi"><div class="v">31%</div><div class="k">为此剔除的样本比例</div></div>
</div>

<div class="card">
<h3>结论</h3>
<p><strong>方向是对的，但收益很小。</strong>保护间隔确实筛掉了误差更大的样本
（事件 ±10 s 内的 MAE 是 1.067 dB，其余样本 0.945 dB），但这类样本只占 10%，
所以整体 MAE 最多只能改善约 0.012 dB——实测在 ±20 s 时改善 {t20['diff_mean_dB']:.4f} dB，
与这个上限几乎吻合。</p>
<p>代价是剔除 31% 的数据。若只取 ±5 s 或 ±10 s，改善在统计上完全不显著（p = 0.64 / 0.95）。</p>
<p class="sub">换句话说：<strong>清洗能去掉噪声，但不能创造信息。</strong>
这份数据的瓶颈是 RSRP 本身对场强的解释力（Table 1 中 r = 0.53~0.76），
而不是切换处的标签不一致。</p>
</div>

<h2>1　数据处理上的一个前提</h2>
<p>先确认了一件事：<strong>每个数据文件只包含一个 earfcn</strong>（例如
<code>round1_RSRP_earfcn100.txt</code> 里 3,494 条全是 100）。所以论文按频段分别计算场强统计量
（EARFCN100 用第三四分位、其余用最大值）这件事，本身不存在"标签定义随频段漂移"的问题。</p>
<p>真正的切换发生在<strong>同一频段内的服务小区之间</strong>：表现为 1 s 内 RSRP 突变 &gt;10 dB，
或者时间轴上出现几秒到几十秒的中断（手机短暂驻留到别的频段/小区，本频段没有记录）。
这类时刻前后，7 点窗口会跨过两段不同信道的样本，特征与标签的对应关系被破坏——
这正是保护间隔要处理的对象。共检测到 <strong>485 次</strong>这样的事件。</p>

<h2>2　结果</h2>
<table class="tbl"><thead><tr><th>保护间隔</th><th>样本数</th><th>保留比例</th>
<th>测试 MAE</th><th>测试 R²</th><th>训练 MAE</th><th>ΔMAE</th><th>胜出划分</th><th>Wilcoxon p</th>
</tr></thead><tbody>{rows}</tbody></table>
<p class="sub">ΔMAE = 基线 − 该设置，正值表示误差变小；胜出划分是 8 个随机种子里改善的个数。</p>

{img(FIG/'fig_guard.png', '(a) 保护间隔越长，剔除的数据越多，MAE 只在 ±20 s 时有 0.013 dB 的改善；<br/>(b) 切换事件附近的样本误差确实更大，但数量只占 10%。')}

<h2>3　分频段看：改善集中在最"安静"的频段</h2>
<table class="tbl"><thead><tr><th>测试 MAE</th><th>无保护间隔</th><th>±5 s</th><th>±10 s</th><th>±20 s</th></tr></thead>
<tbody>{band_rows}</tbody></table>
<table class="tbl"><thead><tr><th>组内 R²</th><th>无保护间隔</th><th>±5 s</th><th>±10 s</th><th>±20 s</th></tr></thead>
<tbody>{band_r2_rows}</tbody></table>
<p>EARFCN100 的组内 R² 从 −0.05 提升到 0.23、MAE 从 0.627 降到 0.534——这是四个频段里
唯一明显受益的。原因很直接：该频段场强本身很平稳（σ≈0.86 dB），切换造成的污染相对而言占比最大，
去掉之后模型才勉强能"看见"真实起伏。而 EARFCN1300/38400 反而略微变差（1.245→1.277、0.959→1.013），
因为它们本身起伏大，切换附近的样本并不比别处更"离群"，剔除只是减少了训练数据。</p>

<div class="note">
<strong>为什么不建议采用这个处理？</strong>如果目标是"让标签定义干净"，这个处理在概念上是对的；
但从工程角度，用 31% 的数据换 0.013 dB（1.3%）不划算，而且 ±5/±10 s 的版本毫无统计显著性。
真正能提升精度的方向是增加信息量——把 RSRQ/SINR/RSSI 等多参量纳入（这正是黄祺等 2026 的做法），
或加入频段信息、按频段分别建模。
</div>

<h2>4　复现方式</h2>
<p><code>python guard_experiment.py --guards 0 5 10 20 --seeds 8</code>　
结果在 <code>outputs/guard/</code>，图在 <code>outputs/figures/fig_guard.png</code>。</p>

<footer>实现：<code>src/features.py</code> 中的 <code>detect_switch_events</code> / <code>guard_mask</code>，
<code>build_samples(guard_s=...)</code>；默认 <code>guard_s=0</code>（不启用），不影响论文复现结果。</footer>
</div></body></html>"""
    (C.OUT_DIR / "report_guard.html").write_text(html, encoding="utf-8")
    print("report ->", C.OUT_DIR / "report_guard.html")


if __name__ == "__main__":
    main()

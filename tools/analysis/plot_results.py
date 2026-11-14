# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
"""Draw the README figures from the JSON written by collect_results.py.

    uv run --no-project --with matplotlib --with pandas \
        python tools/analysis/plot_results.py --data results.json --out figures

Expected run names in the JSON: base, sft, grpo_128x8_60, opd_40, scaffold_40, pipeline_scaffold, pipeline_grpo,
the budget pair scaffold_40:opd_40, and the training logs "scaffold" and "opd". Colors: the Scaffold-OPD runs are blue,
OPD is ochre, everything else is neutral gray.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SCAFFOLD = "#3669AB"
OPD = "#B5812E"
NEUTRAL = "#a6a59e"
DPI = 200
BAR_IN = 0.24  # bar thickness in inches
END_RADIUS_PT = 3  # rounded data end of a bar
# One training step of the Scaffold-OPD run produced degenerate rollouts (looping text) and is left out of its curves.
ANOMALY_STEP = 19
# Training time on two A100 80 GB GPUs, in GPU-hours, evaluation excluded. SFT includes generating the teacher data
# (4.5 h) and one epoch (3.9 h); the pipeline adds Scaffold-OPD (7.7 h) and GRPO (10.0 h).
GPU_HOURS = {"sft": 16.8, "grpo_128x8_60": 33.8, "opd_40": 13.2, "scaffold_40": 13.2, "pipeline_grpo": 52.4}
PIPELINE_CUMULATIVE_GPU_HOURS = [None, 16.8, 32.2, 52.4]

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10.5,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


def px_per_unit(ax):
    """Display pixels per data unit along x and y; valid once limits and the axes position are fixed."""
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    return abs(x1 - x0), abs(y1 - y0)


def bar(ax, pos, value, color, horizontal):
    """A bar of BAR_IN thickness growing from 0, square at the baseline and rounded at the data end."""
    ppx, ppy = px_per_unit(ax)
    thick = BAR_IN * DPI / (ppy if horizontal else ppx)
    r_px = END_RADIUS_PT * DPI / 72
    rx, ry = (r_px / ppx, r_px / ppy) if horizontal else (r_px / ppy, r_px / ppx)
    if value <= 2 * rx:
        rect = (0, pos - thick / 2, value, thick) if horizontal else (pos - thick / 2, 0, thick, value)
        ax.add_patch(Rectangle(rect[:2], *rect[2:], color=color, linewidth=0, zorder=2))
        return
    if horizontal:
        ax.add_patch(Rectangle((0, pos - thick / 2), value - rx, thick, color=color, linewidth=0, zorder=2))
        ax.add_patch(
            FancyBboxPatch(
                (value - 2 * rx, pos - thick / 2),
                2 * rx,
                thick,
                boxstyle=f"round,pad=0,rounding_size={rx}",
                mutation_aspect=ry / rx,
                color=color,
                linewidth=0,
                zorder=3,
            )
        )
    else:
        ax.add_patch(Rectangle((pos - thick / 2, 0), thick, value - rx, color=color, linewidth=0, zorder=2))
        ax.add_patch(
            FancyBboxPatch(
                (pos - thick / 2, value - 2 * rx),
                thick,
                2 * rx,
                boxstyle=f"round,pad=0,rounding_size={ry}",
                mutation_aspect=rx / ry,
                color=color,
                linewidth=0,
                zorder=3,
            )
        )


def title(fig, text, y):
    fig.text(0.012, y, text, fontsize=11.5, color=INK, ha="left", va="top", fontweight="bold")


def subtitle(fig, text, y):
    fig.text(0.012, y, text, fontsize=8.5, color=INK2, ha="left", va="top")


def fig_methods(data, out):
    """Every method trained from the same untrained Qwen3-1.7B student."""
    e = data["eval"]
    rows = [("Untrained", "Qwen3-1.7B", e["base"]["avg4"], NEUTRAL)]  # bottom to top
    for name, detail, key, color in (
        ("GRPO", "60 steps x 128 prompts x 8 rollouts", "grpo_128x8_60", NEUTRAL),
        ("SFT", "filtered teacher data, 1 epoch", "sft", NEUTRAL),
        ("OPD", "40 steps x 128 prompts", "opd_40", OPD),
        ("Scaffold-OPD", "40 steps x 128 prompts", "scaffold_40", SCAFFOLD),
        ("Full pipeline", "SFT → Scaffold-OPD → GRPO", "pipeline_grpo", SCAFFOLD),
    ):
        rows.append((name, f"{detail} · {GPU_HOURS[key]:.1f} GPU-h", e[key]["avg4"], color))
    fig, ax = plt.subplots(figsize=(8.8, 3.9), dpi=DPI)
    fig.subplots_adjust(left=0.32, right=0.97, top=0.79, bottom=0.14)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, 51, 10))
    ax.set_xlabel("4-benchmark average accuracy (%)")
    for i, (name, detail, value, color) in enumerate(rows):
        bar(ax, i, value, color, horizontal=True)
        ax.text(value + 0.8, i, f"{value:.2f}", va="center", ha="left", fontsize=9, color=INK)
        ax.text(-0.025, i + 0.13, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9.5)
        ax.text(
            -0.025,
            i - 0.2,
            detail,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.3,
            color=MUTED,
        )
    pair = data["paired"]["scaffold_40-opd_40"]
    x = 52.0
    ax.plot([x - 0.6, x, x, x - 0.6], [3, 3, 4, 4], color=INK2, linewidth=0.9, zorder=4)
    ax.text(x + 0.8, 3.62, f"+{pair['diff']:.2f} ± {pair['se']:.2f}", fontsize=9, va="center", ha="left")
    ax.text(x + 0.8, 3.3, "paired diff.", fontsize=7.5, color=MUTED, va="center", ha="left")
    title(fig, "Same 1.7B student, same data, same evaluation", 0.975)
    subtitle(
        fig,
        "Teacher Qwen3-4B-Instruct-2507, DAPO-Math · average of MATH-500 and AIME 2024/25/26, T=1.0, 32k tokens "
        "· single seed",
        0.905,
    )
    fig.savefig(f"{out}/methods.png", dpi=DPI)
    plt.close(fig)


def fig_pipeline(data, out):
    """The pipeline stage by stage, next to Scaffold-OPD trained directly from the untrained student."""
    e = data["eval"]
    stages = [
        ("Untrained", e["base"]["avg4"]),
        ("SFT", e["sft"]["avg4"]),
        ("+ Scaffold-OPD\n40 steps", e["pipeline_scaffold"]["avg4"]),
        ("+ GRPO 30 steps\non failures", e["pipeline_grpo"]["avg4"]),
    ]
    direct_x = 4.7
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=DPI)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.27)
    ax.set_xlim(-0.6, direct_x + 0.6)
    ax.set_ylim(0, 64)
    ax.set_yticks(range(0, 51, 10))
    ax.set_xticks([])
    ax.set_ylabel("4-benchmark average accuracy (%)")
    prev = None
    for i, (name, value) in enumerate(stages):
        bar(ax, i, value, NEUTRAL, horizontal=False)
        ax.text(i, value + 1.0, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        if prev is not None:
            ax.text(i, value + 5.0, f"+{value - prev:.2f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
        prev = value
        ax.text(i, -2.2, name, ha="center", va="top", fontsize=8.5, linespacing=1.3)
        hours = PIPELINE_CUMULATIVE_GPU_HOURS[i]
        ax.text(
            i,
            -13.5,
            "—" if hours is None else f"{hours:.1f} GPU-h total",
            ha="center",
            va="top",
            fontsize=7.5,
            color=MUTED,
        )
    direct = e["scaffold_40"]["avg4"]
    bar(ax, direct_x, direct, SCAFFOLD, horizontal=False)
    ax.text(direct_x, direct + 1.0, f"{direct:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(
        direct_x, direct + 5.0, f"+{direct - stages[0][1]:.2f}", ha="center", va="bottom", fontsize=7.5, color=MUTED
    )
    ax.text(direct_x, -2.2, "Scaffold-OPD only\n40 steps", ha="center", va="top", fontsize=8.5, linespacing=1.3)
    ax.text(direct_x, -13.5, f"{GPU_HOURS['scaffold_40']:.1f} GPU-h", ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.axvline((3 + direct_x) / 2, color=GRID, linewidth=0.8, zorder=1)
    pair = data["paired"]["pipeline_grpo-scaffold_40"]
    top = max(stages[-1][1], direct) + 10.0
    ax.plot([3, 3, direct_x, direct_x], [top - 0.8, top, top, top - 0.8], color=INK2, linewidth=0.9)
    ax.text(
        (3 + direct_x) / 2,
        top + 0.6,
        f"{pair['diff']:+.2f} ± {pair['se']:.2f} at about 4x the GPU-hours",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.legend(
        handles=[
            Patch(color=NEUTRAL, label="Pipeline SFT → Scaffold-OPD → GRPO"),
            Patch(color=SCAFFOLD, label="Scaffold-OPD only"),
        ],
        loc="upper left",
        ncols=2,
        bbox_to_anchor=(0.085, 0.86),
        handlelength=1.0,
    )
    title(fig, "Every stage adds accuracy; the full pipeline ties Scaffold-OPD alone", 0.975)
    subtitle(
        fig, "Qwen3-1.7B from the untrained model · MATH-500 and AIME 2024/25/26, T=1.0, 32k tokens · single seed", 0.9
    )
    fig.savefig(f"{out}/pipeline.png", dpi=DPI)
    plt.close(fig)


def log2_budget_axis(ax):
    ax.set_xscale("log", base=2)
    ticks = [1024, 2048, 4096, 8192, 16384, 32768]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1024}k" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(1024 / 1.08, 32768 * 1.08)
    ax.set_xlabel("generation budget at evaluation (tokens)")


def end_dot(ax, x, y, color):
    ax.plot(
        [x],
        [y],
        marker="o",
        markersize=7,
        color=color,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        linestyle="none",
        zorder=4,
    )


def fig_budget(data, out):
    """Scaffold-OPD vs OPD (both step 40) as the evaluation token budget grows."""
    b = data["budget"]
    budgets = b["budgets"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), dpi=DPI)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.74, bottom=0.17, wspace=0.28)
    ax = axes[0]
    for name, color in (("opd_40", OPD), ("scaffold_40", SCAFFOLD)):
        ax.plot(budgets, b[name]["avg4"], color=color, linewidth=2)
        end_dot(ax, budgets[-1], b[name]["avg4"][-1], color)
    ax.text(
        budgets[-1] * 0.93,
        b["scaffold_40"]["avg4"][-1] + 1.5,
        f"{b['scaffold_40']['avg4'][-1]:.1f}",
        ha="right",
        fontsize=8.5,
    )
    ax.text(
        budgets[-1] * 0.93, b["opd_40"]["avg4"][-1] - 3.2, f"{b['opd_40']['avg4'][-1]:.1f}", ha="right", fontsize=8.5
    )
    ax.set_ylim(10, 52)
    ax.set_title("4-benchmark average accuracy (%)")
    log2_budget_axis(ax)

    ax = axes[1]
    diff = b["scaffold_40-opd_40"]["diff"]
    half = [1.96 * s for s in b["scaffold_40-opd_40"]["se"]]
    ax.fill_between(
        budgets,
        [d - h for d, h in zip(diff, half, strict=True)],
        [d + h for d, h in zip(diff, half, strict=True)],
        color=SCAFFOLD,
        alpha=0.12,
        linewidth=0,
    )
    ax.axhline(0, color=AXIS, linewidth=0.9)
    ax.plot(budgets, diff, color=SCAFFOLD, linewidth=2)
    end_dot(ax, budgets[-1], diff[-1], SCAFFOLD)
    for target, text, fx, dy, ha in (
        (4096, "4k: {:+.1f}", 1.0, -1.7, "center"),
        (8192, "8k: {:+.1f}", 1 / 1.1, 0.9, "right"),
        (16384, "16k: {:+.1f}", 1.1, -1.0, "left"),
    ):
        i = budgets.index(target)
        ax.plot([target], [diff[i]], marker="o", markersize=4.5, color=SCAFFOLD, linestyle="none", zorder=4)
        ax.text(target * fx, diff[i] + dy, text.format(diff[i]), ha=ha, fontsize=8, color=INK2)
    ax.text(budgets[-1] * 0.93, diff[-1] + 1.1, f"32k: {diff[-1]:+.1f}", ha="right", fontsize=8.5)
    ax.set_ylim(-7, 9)
    ax.set_title("Scaffold-OPD − OPD (paired, 95% band)")
    log2_budget_axis(ax)

    ax = axes[2]
    for name, color in (("opd_40", OPD), ("scaffold_40", SCAFFOLD)):
        ax.plot(budgets, b[name]["aime_finished"], color=color, linewidth=2)
        end_dot(ax, budgets[-1], b[name]["aime_finished"][-1], color)
    i = budgets.index(16384)
    ax.axvline(16384, color=GRID, linewidth=0.9, zorder=1)
    ax.text(16384 * 0.92, 8, "training response cap 16k", ha="right", fontsize=7.5, color=MUTED)
    for name, dy in (("opd_40", -1), ("scaffold_40", -2)):
        ax.text(
            16384 * 1.08,
            b[name]["aime_finished"][i] + dy,
            f"{b[name]['aime_finished'][i]:.0f}%",
            ha="left",
            va="top",
            fontsize=8,
        )
    ax.set_ylim(0, 105)
    ax.set_title("AIME responses finished within budget (%)")
    log2_budget_axis(ax)

    fig.legend(
        handles=[
            Line2D([], [], color=SCAFFOLD, linewidth=2, label="Scaffold-OPD, step 40"),
            Line2D([], [], color=OPD, linewidth=2, label="OPD, step 40"),
        ],
        loc="upper right",
        ncols=2,
        bbox_to_anchor=(0.97, 0.985),
    )
    title(fig, "The advantage of Scaffold-OPD grows with the reasoning budget", 0.975)
    subtitle(
        fig,
        "The same 32k evaluation outputs cut back to each budget (correct only if finished within it) · "
        "Qwen3-1.7B, T=1.0, single seed",
        0.905,
    )
    fig.savefig(f"{out}/budget.png", dpi=DPI)
    plt.close(fig)


def smooth(series):
    return series.rolling(5, center=True, min_periods=3).mean()


def window(series, steps, lo, hi):
    return float(series[(steps >= lo) & (steps <= hi)].mean())


def fig_training(data, out):
    """Training curves of Scaffold-OPD and OPD, 128 prompts per step."""
    runs = {}
    for name in ("scaffold", "opd"):
        df = pd.DataFrame(data["train"][name]).set_index("step").astype(float)
        if name == "scaffold":
            df.loc[ANOMALY_STEP] = float("nan")
            df["teacher_tokens"] = df["relay/teacher_token_ratio"] * df["response_length/mean"]
        runs[name] = df
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), dpi=DPI)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.08, hspace=0.42, wspace=0.2)
    panels = [  # (axes, metric, runs, title, scale)
        (
            axes[0, 0],
            "opd/k1/student_leg_mean",
            ("scaffold", "opd"),
            "Student-written tokens: k1 (reverse-KL estimate)",
            1,
        ),
        (axes[0, 1], "response_length/mean", ("scaffold", "opd"), "Mean training response length (tokens)", 1),
        (axes[1, 0], "relay/teacher_token_ratio", ("scaffold",), "Scaffold-OPD: share of teacher tokens (%)", 100),
        (axes[1, 1], "teacher_tokens", ("scaffold",), "Scaffold-OPD: teacher tokens per response", 1),
    ]
    colors = {"scaffold": SCAFFOLD, "opd": OPD}
    for ax, key, names, label, scale in panels:
        ax.axvspan(ANOMALY_STEP - 0.5, ANOMALY_STEP + 0.5, color=GRID, alpha=0.7, linewidth=0, zorder=0)
        for name in names:
            s = runs[name][key] * scale
            steps = s.index.to_series()
            ax.plot(steps, s, color=colors[name], linewidth=1, alpha=0.3)
            sm = smooth(s)
            ax.plot(steps, sm, color=colors[name], linewidth=2)
            last = sm.dropna()
            end_dot(ax, last.index[-1], last.iloc[-1], colors[name])
        ax.set_title(label)
        ax.set_xlim(0, 62)
    scaffold = runs["scaffold"]
    steps = scaffold.index.to_series()
    ratio = scaffold["relay/teacher_token_ratio"] * 100
    tokens = scaffold["teacher_tokens"]
    length = scaffold["response_length/mean"]
    axes[1, 0].text(
        60,
        ratio.max() * 0.92,
        f"steps 1–5: {window(ratio, steps, 1, 5):.1f}% → steps 36–40: {window(ratio, steps, 36, 40):.1f}%",
        ha="right",
        fontsize=8,
        color=INK2,
    )
    axes[1, 1].text(
        60,
        tokens.max() * 0.97,
        f"steps 1–5: {window(tokens, steps, 1, 5):.0f} → steps 36–40: {window(tokens, steps, 36, 40):.0f}",
        ha="right",
        fontsize=8,
        color=INK2,
    )
    axes[1, 1].set_ylim(0, tokens.max() * 1.08)
    axes[1, 1].text(
        60,
        tokens.max() * 0.86,
        f"response length {window(length, steps, 1, 5):,.0f} → {window(length, steps, 36, 40):,.0f} tokens",
        ha="right",
        fontsize=8,
        color=MUTED,
    )
    axes[0, 1].text(
        ANOMALY_STEP + 1,
        axes[0, 1].get_ylim()[1] * 0.93,
        f"step {ANOMALY_STEP}: degenerate rollouts, left out",
        fontsize=7.5,
        color=MUTED,
    )
    for ax in axes[1]:
        ax.set_xlabel("training step")
    fig.legend(
        handles=[
            Line2D([], [], color=SCAFFOLD, linewidth=2, label="Scaffold-OPD"),
            Line2D([], [], color=OPD, linewidth=2, label="OPD"),
        ],
        loc="upper right",
        ncols=2,
        bbox_to_anchor=(0.97, 0.985),
    )
    title(fig, "During training the teacher takes over less while the student writes longer", 0.975)
    subtitle(
        fig,
        "Qwen3-1.7B, 128 prompts per step, 16k response cap · thin: per step, thick: centered 5-step mean "
        "· single seed",
        0.935,
    )
    fig.savefig(f"{out}/training.png", dpi=DPI)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="JSON written by collect_results.py")
    parser.add_argument("--out", required=True, help="output directory for the PNGs")
    args = parser.parse_args()
    with open(args.data) as f:
        data = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    fig_methods(data, args.out)
    fig_pipeline(data, args.out)
    fig_budget(data, args.out)
    fig_training(data, args.out)


if __name__ == "__main__":
    main()

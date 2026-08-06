"""Compose the thermal transfer evidence layout.

Panel d receives the wider upper-right region because it carries the complete
held-out cohort and task-performance endpoint. Panel b moves to the lower-right
region. Earlier generators and outputs remain unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import figure_6_source_data as source_data
import figure_6_numerical_panels as numerical_panels
import figure_6_endpoint_panel as endpoint_panel
from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    clean_axes,
    panel_label,
    save_figure,
)


FIGURE_DIR = Path(__file__).resolve().parent
OUT_STEM = FIGURE_DIR / "Figure_6_layout"


def numerical_levels_panel(ax: plt.Axes) -> None:
    data = source_data.load_resolution_data()
    preliminary = data["unvalidated_16_to_32"]
    validated = data["validated_32_to_64"]
    stages = np.asarray([3.0, 2.0, 1.0, 0.0])
    preliminary_values = np.asarray(
        [
            preliminary["16x16 estimate"],
            preliminary["32x32 estimate"],
            preliminary["certificate upper bound"],
            preliminary["final HF episode peak"],
        ],
        dtype=float,
    )
    validated_values = np.asarray(
        [
            validated["32x32 estimate"],
            validated["64x64 estimate"],
            validated["certificate upper bound"],
            validated["final HF episode peak"],
        ],
        dtype=float,
    )

    ax.axvspan(2.0, 2.0135, color=COLORS["light_red"], zorder=0)
    ax.axvline(
        2.0,
        color=COLORS["unsafe"],
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=1,
    )

    for values, line_color, outcome_color in [
        (preliminary_values, COLORS["certificate"], COLORS["unsafe"]),
        (validated_values, COLORS["candidate"], COLORS["controller"]),
    ]:
        ax.plot(
            values[:3],
            stages[:3],
            color=line_color,
            linewidth=1.15,
            zorder=2,
        )
        ax.plot(
            values[2:],
            stages[2:],
            color=outcome_color,
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            zorder=2,
        )
        ax.scatter(
            values[:2],
            stages[:2],
            marker="o",
            s=24,
            facecolor="white",
            edgecolor=line_color,
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            values[2],
            stages[2],
            marker="D",
            s=28,
            color=line_color,
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
        ax.scatter(
            values[3],
            stages[3],
            marker="X",
            s=39,
            color=outcome_color,
            edgecolor="white",
            linewidth=0.35,
            zorder=5,
        )

    marker_handles = [
        Line2D(
            [],
            [],
            marker="o",
            markersize=np.sqrt(24),
            markerfacecolor="white",
            markeredgecolor=COLORS["charcoal"],
            markeredgewidth=0.9,
            linestyle="none",
            label="Estimate",
        ),
        Line2D(
            [],
            [],
            marker="D",
            markersize=np.sqrt(28),
            markerfacecolor=COLORS["charcoal"],
            markeredgecolor="white",
            markeredgewidth=0.4,
            linestyle="none",
            label="Upper risk",
        ),
        Line2D(
            [],
            [],
            marker="X",
            markersize=np.sqrt(39),
            markerfacecolor=COLORS["charcoal"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            linestyle="none",
            label="HF outcome",
        ),
    ]
    ax.legend(
        handles=marker_handles,
        loc="lower left",
        bbox_to_anchor=(0.012, 0.022),
        ncol=1,
        fontsize=4.8,
        labelspacing=0.55,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    ax.text(
        preliminary_values[0] - 0.0082,
        2.82,
        r"$16^2\!\rightarrow\!32^2$",
        color=COLORS["certificate"],
        fontsize=5.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        validated_values[0] + 0.0032,
        2.76,
        r"$32^2\!\rightarrow\!64^2$",
        color=COLORS["candidate"],
        fontsize=5.0,
        fontweight="bold",
        ha="left",
        va="center",
    )

    ax.set_xlim(1.944, 2.0135)
    ax.set_ylim(-0.42, 3.48)
    ax.set_xticks([1.95, 1.975, 2.00])
    ax.set_yticks(stages)
    ax.set_yticklabels(
        [
            r"$Z_{\ell}$",
            r"$Z_{\ell+1}$",
            r"$Z^{+}$",
            r"$Z_{\mathrm{HF}}$",
        ]
    )
    ax.set_xlabel("Maximum temperature")
    clean_axes(ax, grid=False)
    panel_label(ax, "a", x=-0.17, y=1.04)


def certification_time_panel(ax: plt.Axes) -> None:
    always, adaptive = source_data.load_validation_cost()
    order = np.argsort(always)
    jitter = np.linspace(-0.045, 0.045, len(always))
    level_x = jitter
    adaptive_x = 1.0 + jitter

    for offset, index in enumerate(order):
        ax.plot(
            [level_x[offset], adaptive_x[offset]],
            [always[index], adaptive[index]],
            color=COLORS["mid_gray"],
            linewidth=0.45,
            alpha=0.28,
            zorder=1,
        )
    ax.scatter(
        level_x,
        always[order],
        s=10,
        color=COLORS["candidate"],
        alpha=0.52,
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )
    ax.scatter(
        adaptive_x,
        adaptive[order],
        s=10,
        color=COLORS["controller"],
        alpha=0.60,
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )

    mean_level = float(always.mean())
    mean_adaptive = float(adaptive.mean())
    reduction = 100.0 * (1.0 - mean_adaptive / mean_level)
    assert math.isclose(reduction, 46.63973667819058)
    ax.scatter(
        [0.0, 1.0],
        [mean_level, mean_adaptive],
        marker="D",
        s=34,
        color=[COLORS["candidate"], COLORS["controller"]],
        edgecolor="white",
        linewidth=0.45,
        zorder=6,
    )

    guide_x = 0.47
    ax.plot(
        [0.0, guide_x],
        [mean_level, mean_level],
        color=COLORS["charcoal"],
        linewidth=0.7,
        linestyle=(0, (2, 2)),
        zorder=4,
    )
    ax.plot(
        [guide_x, 1.0],
        [mean_adaptive, mean_adaptive],
        color=COLORS["charcoal"],
        linewidth=0.7,
        linestyle=(0, (2, 2)),
        zorder=4,
    )
    bracket = FancyArrowPatch(
        (guide_x, mean_adaptive),
        (guide_x, mean_level),
        arrowstyle="<->",
        mutation_scale=7.0,
        linewidth=0.75,
        color=COLORS["charcoal"],
        zorder=5,
    )
    ax.add_patch(bracket)
    ax.text(
        guide_x + 0.07,
        0.5 * (mean_level + mean_adaptive),
        "46.6% lower\nmean",
        fontsize=5.1,
        color=COLORS["charcoal"],
        ha="left",
        va="center",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "pad": 0.6,
            "alpha": 0.88,
        },
        zorder=7,
    )
    ax.text(
        0.98,
        0.96,
        "64/64 trajectories identical",
        transform=ax.transAxes,
        fontsize=5.1,
        color=COLORS["controller"],
        fontweight="bold",
        ha="right",
        va="top",
    )

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0.45, 4.05)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([r"$\mathcal{N}_1$", "Adaptive"])
    ax.set_ylabel("Certification time\n(s per decision)", labelpad=3.0)
    clean_axes(ax, grid=False)
    panel_label(ax, "b", x=-0.18, y=1.04)


def heldout_endpoint_panel(fig: plt.Figure, spec) -> None:
    raw, verified = endpoint_panel.load_heldout_episode_peaks()
    task_data = source_data.load_safety_task_frontier()
    order = np.argsort(raw)
    raw_sorted = raw[order]
    verified_sorted = verified[order]
    rank = np.arange(1, len(raw_sorted) + 1)
    changed = np.abs(raw_sorted - verified_sorted) > 1e-12
    raw_unsafe = raw_sorted > 1.0

    frame = fig.add_subplot(spec)
    frame.set_axis_off()
    panel_label(frame, "d", x=-0.065, y=1.02)
    frame.text(
        0.02,
        1.02,
        "Held-out cohort",
        transform=frame.transAxes,
        fontsize=6.4,
        fontweight="bold",
        color=COLORS["charcoal"],
        ha="left",
        va="bottom",
    )

    inner = spec.subgridspec(
        2,
        1,
        height_ratios=[0.76, 0.24],
        hspace=0.42,
    )
    risk_ax = fig.add_subplot(inner[0, 0])
    task_ax = fig.add_subplot(inner[1, 0])

    risk_ax.axhspan(1.0, 1.015, color=COLORS["light_red"], zorder=0)
    risk_ax.axhline(
        1.0,
        color=COLORS["charcoal"],
        linewidth=0.75,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    risk_ax.plot(
        rank,
        raw_sorted,
        color=COLORS["mid_gray"],
        linewidth=0.9,
        alpha=0.95,
        zorder=2,
    )
    risk_ax.plot(
        rank,
        verified_sorted,
        color=COLORS["controller"],
        linewidth=1.1,
        alpha=0.92,
        zorder=3,
    )
    risk_ax.scatter(
        rank,
        raw_sorted,
        s=5,
        facecolor="white",
        edgecolor=COLORS["mid_gray"],
        linewidth=0.35,
        alpha=0.85,
        zorder=4,
    )
    risk_ax.scatter(
        rank,
        verified_sorted,
        s=5,
        color=COLORS["controller"],
        edgecolor="white",
        linewidth=0.22,
        alpha=0.82,
        zorder=5,
    )

    for x_value, raw_value, verified_value, is_unsafe in zip(
        rank[changed],
        raw_sorted[changed],
        verified_sorted[changed],
        raw_unsafe[changed],
    ):
        risk_ax.plot(
            [x_value, x_value],
            [raw_value, verified_value],
            color=COLORS["unsafe"] if is_unsafe else COLORS["candidate"],
            linewidth=0.8,
            alpha=0.85,
            zorder=6,
        )

    risk_ax.scatter(
        rank[raw_unsafe],
        raw_sorted[raw_unsafe],
        marker="s",
        s=18,
        color=COLORS["unsafe"],
        edgecolor="white",
        linewidth=0.35,
        zorder=8,
    )
    risk_ax.scatter(
        rank[raw_unsafe],
        verified_sorted[raw_unsafe],
        marker="D",
        s=18,
        color=COLORS["controller"],
        edgecolor="white",
        linewidth=0.35,
        zorder=8,
    )
    label_box = {
        "facecolor": "white",
        "edgecolor": "none",
        "pad": 0.55,
        "alpha": 0.88,
    }
    risk_ax.text(
        0.975,
        0.19,
        "Raw  6/128 unsafe",
        transform=risk_ax.transAxes,
        fontsize=5.0,
        color=COLORS["unsafe"],
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=label_box,
    )
    risk_ax.text(
        0.975,
        0.105,
        "Verified  0/128 unsafe",
        transform=risk_ax.transAxes,
        fontsize=5.0,
        color=COLORS["controller"],
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=label_box,
    )

    risk_ax.set_xlim(0.5, 128.5)
    risk_ax.set_ylim(0.946, 1.0145)
    risk_ax.set_xticks([1, 32, 64, 96, 128])
    risk_ax.set_yticks([0.95, 0.975, 1.0, 1.01])
    risk_ax.set_xlabel("Episode rank by raw peak risk")
    risk_ax.set_ylabel("Peak normalized HF risk")
    clean_axes(risk_ax, grid=False)

    task_methods = [
        (
            "Verified",
            task_data["verified continuation"]["task"],
            COLORS["controller"],
            "D",
        ),
        (
            r"$K=5$ only",
            task_data["K=5 continuation only"]["task"],
            COLORS["certificate"],
            "o",
        ),
    ]
    task_y = np.asarray([1.0, 0.0])
    task_ax.axvline(
        100.0,
        color=COLORS["mid_gray"],
        linewidth=0.7,
        linestyle=(0, (2, 2)),
        zorder=0,
    )
    for y_value, (label, value, color, marker) in zip(task_y, task_methods):
        task_ax.hlines(
            y_value,
            90.8,
            value,
            color=color,
            linewidth=1.6,
            alpha=0.58,
            zorder=1,
        )
        task_ax.scatter(
            value,
            y_value,
            marker=marker,
            s=29,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        x_offset = -0.24 if value > 98.0 else 0.22
        task_ax.text(
            value + x_offset,
            y_value,
            f"{value:.2f}",
            fontsize=5.0,
            color=color,
            fontweight="bold",
            ha="right" if x_offset < 0 else "left",
            va="center",
        )

    task_ax.set_xlim(90.6, 100.8)
    task_ax.set_ylim(-0.55, 1.55)
    task_ax.set_xticks([92, 96, 100])
    task_ax.set_yticks(task_y)
    task_ax.set_yticklabels([item[0] for item in task_methods], fontsize=5.0)
    task_ax.set_xlabel("Mean HF task (% of raw)")
    task_ax.tick_params(axis="y", length=0, pad=2)
    clean_axes(task_ax, grid=False)


def main() -> None:
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 5.02))
    outer = fig.add_gridspec(
        2,
        13,
        height_ratios=[0.97, 1.03],
        hspace=0.43,
        wspace=0.64,
        left=0.085,
        right=0.985,
        bottom=0.105,
        top=0.955,
    )
    numerical_levels_panel(fig.add_subplot(outer[0, :5]))
    heldout_endpoint_panel(fig, outer[0, 6:])
    numerical_panels.temporal_risk_panel(fig, outer[1, :7])
    certification_time_panel(fig.add_subplot(outer[1, 8:]))
    save_figure(fig, OUT_STEM.with_suffix(".pdf"))
    print(OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

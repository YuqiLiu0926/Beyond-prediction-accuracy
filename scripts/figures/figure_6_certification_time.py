"""Certification-time comparison for Figure 6.

The mean-time difference is drawn as a dashed shaft with separate arrowheads,
which remains stable in vector and raster exports. Earlier versions remain
unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import figure_6_source_data as source_data
import figure_6_refined_panels as refined_panels
from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    clean_axes,
    panel_label,
    save_figure,
)


FIGURE_DIR = Path(__file__).resolve().parent
OUT_STEM = FIGURE_DIR / "Figure_6_certification_time"


def certification_time_panel(ax: plt.Axes) -> None:
    level, adaptive = source_data.load_validation_cost()
    order = np.argsort(level)
    jitter = np.linspace(-0.045, 0.045, len(level))
    level_x = jitter
    adaptive_x = 1.0 + jitter

    for offset, index in enumerate(order):
        ax.plot(
            [level_x[offset], adaptive_x[offset]],
            [level[index], adaptive[index]],
            color=COLORS["mid_gray"],
            linewidth=0.45,
            alpha=0.28,
            zorder=1,
        )
    ax.scatter(
        level_x,
        level[order],
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

    mean_level = float(level.mean())
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
        zorder=7,
    )

    ax.plot(
        [0.0, 1.0],
        [mean_adaptive, mean_adaptive],
        color=COLORS["charcoal"],
        linewidth=0.7,
        linestyle=(0, (2, 2)),
        zorder=4,
    )

    # Separate shaft and arrowheads prevent dashed FancyArrowPatch distortion.
    arrow_x = 0.0
    arrow_top = mean_level - 0.12
    arrow_bottom = mean_adaptive + 0.10
    ax.plot(
        [arrow_x, arrow_x],
        [arrow_bottom, arrow_top],
        color=COLORS["charcoal"],
        linewidth=0.85,
        linestyle=(0, (2, 2)),
        dash_capstyle="butt",
        zorder=6,
    )
    ax.plot(
        arrow_x,
        arrow_top,
        marker="^",
        markersize=3.7,
        markerfacecolor=COLORS["charcoal"],
        markeredgewidth=0,
        linestyle="none",
        zorder=7,
    )
    ax.plot(
        arrow_x,
        arrow_bottom,
        marker="v",
        markersize=3.7,
        markerfacecolor=COLORS["charcoal"],
        markeredgewidth=0,
        linestyle="none",
        zorder=7,
    )
    ax.text(
        arrow_x,
        mean_adaptive - 0.07,
        "46.6% lower\nmean",
        fontsize=5.1,
        color=COLORS["charcoal"],
        ha="center",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "pad": 0.5,
            "alpha": 0.90,
        },
        zorder=8,
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
    panel_label(ax, "d", x=-0.18, y=1.04)


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
    refined_panels.panel_order.numerical_levels_panel(fig.add_subplot(outer[0, :5]))
    refined_panels.heldout_endpoint_panel(fig, outer[0, 6:])
    refined_panels.temporal_risk_panel(fig, outer[1, :7])
    certification_time_panel(fig.add_subplot(outer[1, 8:]))
    save_figure(fig, OUT_STEM.with_suffix(".pdf"))
    print(OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

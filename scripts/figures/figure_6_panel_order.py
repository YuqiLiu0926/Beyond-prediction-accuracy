"""Arrange the thermal transfer panels and annotations."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

import figure_6_source_data as source_data
import figure_6_layout as layout
from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    clean_axes,
    panel_label,
    save_figure,
)


FIGURE_DIR = Path(__file__).resolve().parent
OUT_STEM = FIGURE_DIR / "Figure_6_panel_order"


def numerical_levels_panel(ax: plt.Axes) -> None:
    layout.numerical_levels_panel(ax)
    blue_level_text = r"$32^2\!\rightarrow\!64^2$"
    for text_artist in ax.texts:
        if text_artist.get_text() == blue_level_text:
            text_artist.set_position((1.9802, 2.84))
            text_artist.set_ha("left")
            text_artist.set_va("center")


def heldout_endpoint_panel(fig: plt.Figure, spec) -> None:
    existing_axes = len(fig.axes)
    layout.heldout_endpoint_panel(fig, spec)
    new_axes = fig.axes[existing_axes:]
    for ax in new_axes:
        for text_artist in ax.texts:
            if text_artist.get_text() == "d":
                text_artist.set_text("b")
            elif text_artist.get_text() == "99.95":
                text_artist.set_x(100.14)
                text_artist.set_ha("left")


def temporal_risk_panel(fig: plt.Figure, spec) -> None:
    _, raw, verified = source_data.load_unsafe_traces()
    labels = [str(index) for index in range(1, 7)]
    frame = fig.add_subplot(spec)
    frame.set_axis_off()
    panel_label(frame, "c", x=-0.065, y=1.02)

    inner = spec.subgridspec(
        1,
        5,
        width_ratios=[1.0, 0.08, 0.045, 0.16, 1.0],
        wspace=0.0,
    )
    raw_ax = fig.add_subplot(inner[0, 0])
    cbar_ax = fig.add_subplot(inner[0, 2])
    verified_ax = fig.add_subplot(inner[0, 4])

    cmap = LinearSegmentedColormap.from_list(
        "thermal_risk_diverging",
        [COLORS["controller"], "#F7F7F7", COLORS["unsafe"]],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=0.96, vcenter=1.0, vmax=1.013)
    image = None
    for ax, matrix, title, color in [
        (raw_ax, raw, "Raw surrogate   6/6 cross", COLORS["unsafe"]),
        (
            verified_ax,
            verified,
            "Verified continuation   0/6 cross",
            COLORS["controller"],
        ),
    ]:
        image = ax.pcolormesh(
            np.arange(matrix.shape[1] + 1),
            np.arange(matrix.shape[0] + 1),
            matrix,
            cmap=cmap,
            norm=norm,
            rasterized=False,
            shading="flat",
        )
        ax.set_title(title, fontsize=5.8, color=color, fontweight="bold", pad=3)
        ax.set_xlim(0, matrix.shape[1])
        ax.set_ylim(matrix.shape[0], 0)
        ax.set_xticks([0.5, 10.5, 20.5])
        ax.set_xticklabels(["0", "10", "20"])
        ax.set_xlabel("Closed-loop step")
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    raw_ax.set_yticklabels(labels, fontsize=5.2)
    raw_ax.set_ylabel("Episode")
    verified_ax.set_yticklabels([])
    assert image is not None
    colorbar = fig.colorbar(image, cax=cbar_ax, extend="min")
    colorbar.set_ticks([0.96, 0.98, 1.00, 1.01])
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.tick_params(
        axis="y",
        labelleft=True,
        labelright=False,
        labelsize=4.8,
        width=0.45,
        length=2.0,
        pad=1.5,
    )
    colorbar.outline.set_linewidth(0.45)
    colorbar.ax.set_title("HF risk", fontsize=5.0, pad=3)


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
    arrow_top = mean_level - 0.075
    difference_arrow = FancyArrowPatch(
        (0.0, mean_adaptive + 0.02),
        (0.0, arrow_top),
        arrowstyle="<->",
        mutation_scale=7.0,
        linewidth=0.8,
        linestyle=(0, (2, 2)),
        color=COLORS["charcoal"],
        zorder=6,
    )
    ax.add_patch(difference_arrow)
    ax.text(
        0.08,
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
    numerical_levels_panel(fig.add_subplot(outer[0, :5]))
    heldout_endpoint_panel(fig, outer[0, 6:])
    temporal_risk_panel(fig, outer[1, :7])
    certification_time_panel(fig.add_subplot(outer[1, 8:]))
    save_figure(fig, OUT_STEM.with_suffix(".pdf"))
    print(OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

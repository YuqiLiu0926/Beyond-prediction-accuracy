"""Thermal endpoint and time-resolved risk panels for Figure 6.

The central colorbar in panel c occupies a dedicated layout region so that
the bar, tick labels and title remain between the two heatmaps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import figure_6_source_data as source_data
import figure_6_endpoint_panel as endpoint_panel
import figure_6_panel_order as panel_order
from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    clean_axes,
    panel_label,
    save_figure,
)


FIGURE_DIR = Path(__file__).resolve().parent
OUT_STEM = FIGURE_DIR / "Figure_6_refined_panels"


def heldout_endpoint_panel(fig: plt.Figure, spec) -> None:
    """Show held-out safety endpoints and task retention without label overlap."""
    raw, verified = endpoint_panel.load_heldout_episode_peaks()
    task_data = source_data.load_safety_task_frontier()
    order = np.argsort(raw)
    raw_sorted = raw[order]
    verified_sorted = verified[order]
    rank = np.arange(1, len(raw_sorted) + 1)
    changed = np.abs(raw_sorted - verified_sorted) > 1e-12
    raw_unsafe = raw_sorted > 1.0

    assert len(raw_sorted) == 128
    assert int(raw_unsafe.sum()) == 6
    assert int((verified_sorted > 1.0).sum()) == 0

    frame = fig.add_subplot(spec)
    frame.set_axis_off()
    panel_label(frame, "b", x=-0.065, y=1.02)
    frame.text(
        0.02,
        1.02,
        "Held-out validation set",
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
        height_ratios=[0.70, 0.30],
        hspace=0.62,
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
    risk_ax.set_xlabel("Episode rank by raw peak risk", labelpad=2.0)
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
            r"$K=5$",
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
        if label == "Verified":
            text_x = value + 0.20
            text_ha = "left"
        else:
            text_x = value + 0.22
            text_ha = "left"
        task_ax.text(
            text_x,
            y_value,
            f"{value:.2f}",
            fontsize=5.0,
            color=color,
            fontweight="bold",
            ha=text_ha,
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


def temporal_risk_panel(fig: plt.Figure, spec) -> None:
    """Place the shared risk scale in a dedicated region between heatmaps."""
    _, raw, verified = source_data.load_unsafe_traces()
    labels = [str(index) for index in range(1, 7)]
    frame = fig.add_subplot(spec)
    frame.set_axis_off()
    panel_label(frame, "c", x=-0.065, y=1.02)

    inner = spec.subgridspec(
        1,
        3,
        width_ratios=[1.0, 0.28, 1.0],
        wspace=0.06,
    )
    raw_ax = fig.add_subplot(inner[0, 0])
    colorbar_frame = fig.add_subplot(inner[0, 1])
    verified_ax = fig.add_subplot(inner[0, 2])
    colorbar_frame.set_axis_off()

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
    cbar_ax = colorbar_frame.inset_axes([0.16, 0.035, 0.22, 0.90])
    colorbar = fig.colorbar(image, cax=cbar_ax, extend="min")
    colorbar.set_ticks([0.96, 0.98, 1.00, 1.01])
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.tick_params(
        axis="y",
        labelleft=False,
        labelright=True,
        labelsize=4.8,
        width=0.45,
        length=2.0,
        pad=1.5,
    )
    colorbar.outline.set_linewidth(0.45)
    colorbar_frame.text(
        0.50,
        1.015,
        "HF risk",
        transform=colorbar_frame.transAxes,
        fontsize=5.0,
        ha="center",
        va="bottom",
    )


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
    panel_order.numerical_levels_panel(fig.add_subplot(outer[0, :5]))
    heldout_endpoint_panel(fig, outer[0, 6:])
    temporal_risk_panel(fig, outer[1, :7])
    panel_order.certification_time_panel(fig.add_subplot(outer[1, 8:]))
    save_figure(fig, OUT_STEM.with_suffix(".pdf"))
    print(OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

"""Numerical-level, certification-cost and outcome panels for Figure 6."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

import figure_6_source_data as source_data
from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    clean_axes,
    panel_label,
    save_figure,
)


FIGURE_DIR = Path(__file__).resolve().parent
OUT_STEM = FIGURE_DIR / "Figure_6_numerical_panels"


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
        ]
    )
    validated_values = np.asarray(
        [
            validated["32x32 estimate"],
            validated["64x64 estimate"],
            validated["certificate upper bound"],
            validated["final HF episode peak"],
        ]
    )

    ax.axvspan(2.0, 2.014, color=COLORS["light_red"], zorder=0)
    ax.axvline(
        2.0,
        color=COLORS["unsafe"],
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    ax.plot(
        preliminary_values[:3],
        stages[:3],
        color=COLORS["certificate"],
        linewidth=1.2,
        zorder=2,
    )
    ax.plot(
        validated_values[:3],
        stages[:3],
        color=COLORS["candidate"],
        linewidth=1.2,
        zorder=2,
    )
    ax.plot(
        preliminary_values[2:],
        stages[2:],
        color=COLORS["unsafe"],
        linewidth=0.95,
        linestyle=(0, (2, 2)),
        zorder=2,
    )
    ax.plot(
        validated_values[2:],
        stages[2:],
        color=COLORS["controller"],
        linewidth=0.95,
        linestyle=(0, (2, 2)),
        zorder=2,
    )

    ax.scatter(
        preliminary_values[:2],
        stages[:2],
        s=25,
        facecolor="white",
        edgecolor=COLORS["certificate"],
        linewidth=0.9,
        zorder=3,
    )
    ax.scatter(
        validated_values[:2],
        stages[:2],
        s=25,
        facecolor="white",
        edgecolor=COLORS["candidate"],
        linewidth=0.9,
        zorder=3,
    )
    ax.scatter(
        preliminary_values[2],
        stages[2],
        marker="D",
        s=29,
        color=COLORS["certificate"],
        edgecolor="white",
        linewidth=0.4,
        zorder=4,
    )
    ax.scatter(
        validated_values[2],
        stages[2],
        marker="D",
        s=29,
        color=COLORS["candidate"],
        edgecolor="white",
        linewidth=0.4,
        zorder=4,
    )
    ax.scatter(
        preliminary_values[3],
        stages[3],
        marker="X",
        s=42,
        color=COLORS["unsafe"],
        linewidth=0.55,
        zorder=5,
    )
    ax.scatter(
        validated_values[3],
        stages[3],
        marker="o",
        s=34,
        color=COLORS["controller"],
        edgecolor="white",
        linewidth=0.4,
        zorder=5,
    )

    ax.set_xlim(1.944, 2.014)
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
    handles = [
        Line2D(
            [],
            [],
            color=COLORS["certificate"],
            marker="o",
            markerfacecolor="white",
            label=r"Preliminary $16^2\!\rightarrow\!32^2$",
        ),
        Line2D(
            [],
            [],
            color=COLORS["candidate"],
            marker="o",
            markerfacecolor="white",
            label=r"Validated $32^2\!\rightarrow\!64^2$",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.06),
        ncol=2,
        columnspacing=0.8,
        handlelength=1.25,
        handletextpad=0.35,
        fontsize=5.2,
    )
    clean_axes(ax, grid=False)
    panel_label(ax, "a", x=-0.13, y=1.04)


def certification_time_panel(ax: plt.Axes) -> None:
    always, adaptive = source_data.load_validation_cost()
    order = np.argsort(always)
    jitter = np.linspace(-0.045, 0.045, len(always))
    always_x = jitter
    adaptive_x = 1.0 + jitter

    for offset, index in enumerate(order):
        ax.plot(
            [always_x[offset], adaptive_x[offset]],
            [always[index], adaptive[index]],
            color=COLORS["mid_gray"],
            linewidth=0.45,
            alpha=0.30,
            zorder=1,
        )
    ax.scatter(
        always_x,
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

    mean_always = float(always.mean())
    mean_adaptive = float(adaptive.mean())
    reduction = 100.0 * (1.0 - mean_adaptive / mean_always)
    assert math.isclose(reduction, 46.63973667819058)
    ax.plot(
        [0.0, 1.0],
        [mean_always, mean_adaptive],
        color=COLORS["charcoal"],
        linewidth=1.3,
        zorder=4,
    )
    ax.scatter(
        [0.0, 1.0],
        [mean_always, mean_adaptive],
        marker="D",
        s=35,
        color=[COLORS["candidate"], COLORS["controller"]],
        edgecolor="white",
        linewidth=0.45,
        zorder=5,
    )
    ax.text(
        0.52,
        2.28,
        r"$-46.6\%$ mean",
        fontsize=5.4,
        color=COLORS["charcoal"],
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.90},
        zorder=6,
    )
    ax.text(
        0.98,
        0.96,
        "64/64 trajectories identical",
        transform=ax.transAxes,
        fontsize=5.2,
        color=COLORS["controller"],
        fontweight="bold",
        ha="right",
        va="top",
    )

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0.45, 4.05)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([r"Always $\mathcal{N}_1$", "Adaptive"])
    ax.set_ylabel("Certification time (s per decision)")
    clean_axes(ax, grid=False)
    panel_label(ax, "b", x=-0.24, y=1.04)


def temporal_risk_panel(fig: plt.Figure, spec) -> None:
    _, raw, verified = source_data.load_unsafe_traces()
    labels = [str(index) for index in range(1, 7)]
    frame = fig.add_subplot(spec)
    frame.set_axis_off()
    panel_label(frame, "c", x=-0.065, y=1.02)

    inner = spec.subgridspec(
        1,
        3,
        width_ratios=[1.0, 0.045, 1.0],
        wspace=0.20,
    )
    raw_ax = fig.add_subplot(inner[0, 0])
    cbar_ax = fig.add_subplot(inner[0, 1])
    verified_ax = fig.add_subplot(inner[0, 2])

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
    colorbar.ax.tick_params(labelsize=4.8, width=0.45, length=2.0)
    colorbar.outline.set_linewidth(0.45)
    colorbar.ax.set_title("HF risk", fontsize=5.0, pad=3)


def compact_outcomes_panel(fig: plt.Figure, spec) -> None:
    data = source_data.load_safety_task_frontier()
    methods = [
        ("raw surrogate", "Raw surrogate", COLORS["unsafe"], "s"),
        (
            "verified continuation",
            "Verified continuation",
            COLORS["controller"],
            "D",
        ),
        (
            "K=5 continuation only",
            r"$K=5$ continuation only",
            COLORS["certificate"],
            "o",
        ),
    ]
    y = np.asarray([2.0, 1.0, 0.0])
    task = np.asarray([data[key]["task"] for key, _, _, _ in methods])
    unsafe = np.asarray([data[key]["unsafe"] for key, _, _, _ in methods])

    frame = fig.add_subplot(spec)
    frame.set_axis_off()
    panel_label(frame, "d", x=-0.08, y=1.02)
    frame.text(
        0.00,
        1.025,
        "Held-out outcomes",
        transform=frame.transAxes,
        fontsize=6.1,
        fontweight="bold",
        color=COLORS["charcoal"],
        ha="left",
        va="bottom",
    )
    frame.text(
        1.00,
        1.025,
        r"$n=128$",
        transform=frame.transAxes,
        fontsize=5.1,
        color=COLORS["unavailable"],
        ha="right",
        va="bottom",
    )

    inner = spec.subgridspec(
        1,
        3,
        width_ratios=[0.92, 1.18, 0.78],
        wspace=0.16,
    )
    label_ax = fig.add_subplot(inner[0, 0])
    task_ax = fig.add_subplot(inner[0, 1])
    unsafe_ax = fig.add_subplot(inner[0, 2], sharey=task_ax)
    label_ax.set_axis_off()
    label_ax.set_xlim(0.0, 1.0)
    label_ax.set_ylim(-0.55, 2.55)
    for index, (_, label, color, _) in enumerate(methods):
        label_ax.text(
            0.98,
            y[index],
            label,
            fontsize=4.9,
            color=color,
            ha="right",
            va="center",
        )
    for ax in (task_ax, unsafe_ax):
        ax.axhspan(0.66, 1.34, color=COLORS["light_green"], zorder=0)

    for index, (_, _, color, marker) in enumerate(methods):
        task_ax.hlines(
            y[index],
            90.5,
            task[index],
            color=color,
            linewidth=1.05,
            alpha=0.58,
            zorder=1,
        )
        task_ax.scatter(
            task[index],
            y[index],
            marker=marker,
            s=36,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        task_offset = -0.32 if task[index] > 98.0 else 0.25
        task_ax.text(
            task[index] + task_offset,
            y[index] + 0.18,
            f"{task[index]:.2f}",
            fontsize=5.0,
            color=color,
            ha="right" if task_offset < 0 else "left",
            va="bottom",
        )

        unsafe_ax.hlines(
            y[index],
            0.0,
            unsafe[index],
            color=color,
            linewidth=1.05,
            alpha=0.58,
            zorder=1,
        )
        unsafe_ax.scatter(
            unsafe[index],
            y[index],
            marker=marker,
            s=36,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        unsafe_ax.text(
            unsafe[index] + 0.28,
            y[index] + 0.18,
            f"{int(unsafe[index])}",
            fontsize=5.0,
            color=color,
            ha="left",
            va="bottom",
        )

    task_ax.axvline(
        100.0,
        color=COLORS["mid_gray"],
        linewidth=0.65,
        linestyle=(0, (2, 2)),
        zorder=0,
    )
    task_ax.set_xlim(90.4, 101.0)
    task_ax.set_xticks([92, 96, 100])
    task_ax.set_xlabel("HF task (% of raw)")
    task_ax.set_yticks(y)
    task_ax.tick_params(axis="y", left=False, labelleft=False)

    unsafe_ax.set_xlim(-0.35, 6.85)
    unsafe_ax.set_xticks([0, 3, 6])
    unsafe_ax.set_xlabel("Unsafe episodes")
    unsafe_ax.tick_params(axis="y", left=False, labelleft=False)
    for ax in (task_ax, unsafe_ax):
        ax.set_ylim(-0.55, 2.55)
        clean_axes(ax, grid=False)


def main() -> None:
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 5.02))
    outer = fig.add_gridspec(
        2,
        12,
        height_ratios=[0.93, 1.07],
        hspace=0.43,
        wspace=0.82,
        left=0.10,
        right=0.985,
        bottom=0.105,
        top=0.955,
    )
    numerical_levels_panel(fig.add_subplot(outer[0, :7]))
    certification_time_panel(fig.add_subplot(outer[0, 7:]))
    temporal_risk_panel(fig, outer[1, :7])
    compact_outcomes_panel(fig, outer[1, 7:])
    save_figure(fig, OUT_STEM.with_suffix(".pdf"))
    print(OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

"""Load the thermal transfer source data used in Figure 6.

The figure uses the existing Figure 6 source-data tables. It gives the
held-out safety--task result the largest visual weight, while retaining the
numerical-bias mechanism, paired validation cost and trajectory-level evidence.
Earlier generators and outputs remain unchanged.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    clean_axes,
    panel_label,
    save_figure,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "outputs" / "figures"
SOURCE_DIR = ROOT / "data" / "source_data" / "csv"
OUT_STEM = FIGURE_DIR / "Figure_6_data_check"


def read_csv(name: str) -> list[dict[str, str]]:
    with (SOURCE_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_resolution_data() -> dict[str, dict[str, float]]:
    rows = read_csv("SourceData_Fig6_thermal_resolution_ladder.csv")
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        grouped.setdefault(row["numerical_check"], {})[row["quantity"]] = float(
            row["maximum_temperature"]
        )
    assert set(grouped) == {"unvalidated_16_to_32", "validated_32_to_64"}
    assert math.isclose(
        grouped["unvalidated_16_to_32"]["certificate upper bound"],
        1.9886841095285,
    )
    assert math.isclose(
        grouped["validated_32_to_64"]["certificate upper bound"],
        2.0082467959778656,
    )
    return grouped


def load_validation_cost() -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv("SourceData_Fig6_thermal_validation_cost.csv")
    always = np.asarray(
        [float(row["always_fine_seconds_per_decision"]) for row in rows], dtype=float
    )
    adaptive = np.asarray(
        [float(row["tiered_seconds_per_decision"]) for row in rows], dtype=float
    )
    agreement = np.asarray(
        [int(row["executed_trajectory_identical"]) for row in rows], dtype=int
    )
    assert len(rows) == 64
    assert int(agreement.sum()) == 64
    assert math.isclose(float(always.mean()), 1.9640574145516438)
    assert math.isclose(float(adaptive.mean()), 1.0480262081962792)
    return always, adaptive


def load_unsafe_traces() -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = read_csv("SourceData_Fig6_thermal_unsafe_traces.csv")
    seed_names = {20261031: "S1", 20261032: "S2"}
    grouped: dict[tuple[int, int, str], list[tuple[int, float]]] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["case_index"]), row["controller"])
        grouped.setdefault(key, []).append(
            (int(row["step"]), float(row["normalized_hf_risk"]))
        )

    cases = sorted({(seed, case) for seed, case, _ in grouped})
    labels = [f"{seed_names[seed]} / E{case}" for seed, case in cases]
    raw = []
    verified = []
    for seed, case in cases:
        raw_row = sorted(grouped[(seed, case, "raw")])
        verified_row = sorted(grouped[(seed, case, "verified_continuation")])
        assert [step for step, _ in raw_row] == list(range(21))
        assert [step for step, _ in verified_row] == list(range(21))
        raw.append([value for _, value in raw_row])
        verified.append([value for _, value in verified_row])

    raw_array = np.asarray(raw, dtype=float)
    verified_array = np.asarray(verified, dtype=float)
    assert raw_array.shape == verified_array.shape == (6, 21)
    assert int(np.sum(np.max(raw_array, axis=1) > 1.0)) == 6
    assert int(np.sum(np.max(verified_array, axis=1) > 1.0)) == 0
    return labels, raw_array, verified_array


def load_safety_task_frontier() -> dict[str, dict[str, float]]:
    rows = read_csv("SourceData_Fig6_thermal_safety_task_frontier.csv")
    data = {
        row["controller"]: {
            "episodes": float(row["episodes"]),
            "unsafe": float(row["unsafe_episodes"]),
            "unsafe_rate": float(row["unsafe_rate_percent"]),
            "task": float(row["mean_hf_task_fraction_of_raw_percent"]),
        }
        for row in rows
    }
    assert set(data) == {
        "raw surrogate",
        "verified continuation",
        "K=5 continuation only",
    }
    assert data["raw surrogate"]["unsafe"] == 6
    assert data["verified continuation"]["unsafe"] == 0
    assert math.isclose(data["verified continuation"]["task"], 99.94815446836206)
    return data


def decision_chain_panel(ax: plt.Axes) -> None:
    data = load_resolution_data()
    preliminary = data["unvalidated_16_to_32"]
    validated = data["validated_32_to_64"]
    y_preliminary, y_validated = 1.0, 0.0

    ax.axvspan(2.0, 2.012, color=COLORS["light_red"], zorder=0)
    ax.axvline(
        2.0,
        color=COLORS["unsafe"],
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=1,
    )

    pre_estimates = [
        preliminary["16x16 estimate"],
        preliminary["32x32 estimate"],
    ]
    pre_upper = preliminary["certificate upper bound"]
    pre_hf = preliminary["final HF episode peak"]
    val_estimates = [
        validated["32x32 estimate"],
        validated["64x64 estimate"],
    ]
    val_upper = validated["certificate upper bound"]
    val_hf = validated["final HF episode peak"]

    ax.plot(
        [*pre_estimates, pre_upper],
        [y_preliminary] * 3,
        color=COLORS["certificate"],
        linewidth=1.15,
        zorder=2,
    )
    ax.plot(
        [*val_estimates, val_upper],
        [y_validated] * 3,
        color=COLORS["candidate"],
        linewidth=1.15,
        zorder=2,
    )
    ax.scatter(
        pre_estimates,
        [y_preliminary] * 2,
        s=24,
        facecolor="white",
        edgecolor=COLORS["certificate"],
        linewidth=0.9,
        zorder=4,
    )
    ax.scatter(
        val_estimates,
        [y_validated] * 2,
        s=24,
        facecolor="white",
        edgecolor=COLORS["candidate"],
        linewidth=0.9,
        zorder=4,
    )
    ax.scatter(
        [pre_upper],
        [y_preliminary],
        marker="D",
        s=30,
        color=COLORS["certificate"],
        edgecolor="white",
        linewidth=0.4,
        zorder=5,
    )
    ax.scatter(
        [val_upper],
        [y_validated],
        marker="D",
        s=30,
        color=COLORS["candidate"],
        edgecolor="white",
        linewidth=0.4,
        zorder=5,
    )

    ax.add_patch(
        FancyArrowPatch(
            (pre_upper, y_preliminary),
            (pre_hf, y_preliminary),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            color=COLORS["unsafe"],
            zorder=3,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (val_upper, y_validated),
            (val_hf, y_validated),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            color=COLORS["controller"],
            connectionstyle="arc3,rad=-0.12",
            zorder=3,
        )
    )
    ax.scatter(
        [pre_hf],
        [y_preliminary],
        marker="X",
        s=43,
        color=COLORS["unsafe"],
        linewidth=0.55,
        zorder=6,
    )
    ax.scatter(
        [val_hf],
        [y_validated],
        marker="o",
        s=34,
        color=COLORS["controller"],
        edgecolor="white",
        linewidth=0.4,
        zorder=6,
    )

    for x, label in zip(pre_estimates, [r"$16^2$", r"$32^2$"]):
        ax.text(
            x,
            y_preliminary - 0.17,
            label,
            fontsize=5.0,
            color=COLORS["unavailable"],
            ha="center",
            va="top",
        )
    for x, label in zip(val_estimates, [r"$32^2$", r"$64^2$"]):
        ax.text(
            x,
            y_validated + 0.17,
            label,
            fontsize=5.0,
            color=COLORS["unavailable"],
            ha="center",
            va="bottom",
        )

    ax.annotate(
        "upper 1.9887\naccept",
        (pre_upper, y_preliminary),
        xytext=(-4, 11),
        textcoords="offset points",
        fontsize=5.0,
        color=COLORS["certificate"],
        ha="right",
        va="bottom",
    )
    ax.annotate(
        "HF 2.0076\nunsafe",
        (pre_hf, y_preliminary),
        xytext=(-2, -10),
        textcoords="offset points",
        fontsize=5.0,
        color=COLORS["unsafe"],
        ha="right",
        va="top",
    )
    ax.annotate(
        "upper 2.0082\nreject",
        (val_upper, y_validated),
        xytext=(-2, -10),
        textcoords="offset points",
        fontsize=5.0,
        color=COLORS["candidate"],
        ha="right",
        va="top",
    )
    ax.annotate(
        "retained HF 1.9890\nsafe",
        (val_hf, y_validated),
        xytext=(3, 11),
        textcoords="offset points",
        fontsize=5.0,
        color=COLORS["controller"],
        ha="left",
        va="bottom",
    )

    ax.set_xlim(1.948, 2.012)
    ax.set_ylim(-0.48, 1.48)
    ax.set_xticks([1.95, 1.975, 2.00])
    ax.set_yticks([])
    ax.text(
        1.949,
        y_preliminary + 0.22,
        r"Preliminary $16^2\!\rightarrow\!32^2$",
        fontsize=5.3,
        color=COLORS["certificate"],
        ha="left",
        va="bottom",
    )
    ax.text(
        1.949,
        y_validated + 0.22,
        r"Validated $32^2\!\rightarrow\!64^2$",
        fontsize=5.3,
        color=COLORS["candidate"],
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Maximum temperature")
    handles = [
        Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="white",
            markeredgecolor=COLORS["unavailable"], label="solver estimate"
        ),
        Line2D(
            [], [], marker="D", linestyle="none", color=COLORS["unavailable"],
            label="upper risk"
        ),
        Line2D(
            [], [], marker="X", linestyle="none", color=COLORS["unsafe"],
            label="HF outcome"
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.00, 1.13),
        ncol=3,
        columnspacing=0.65,
        handletextpad=0.25,
        fontsize=5.0,
    )
    clean_axes(ax, grid=False)
    panel_label(ax, "a", x=-0.14, y=1.08)


def validation_cost_panel(ax: plt.Axes) -> None:
    always, adaptive = load_validation_cost()
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
            alpha=0.32,
            zorder=1,
        )
    ax.scatter(
        always_x,
        always[order],
        s=10,
        color=COLORS["candidate"],
        alpha=0.55,
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )
    ax.scatter(
        adaptive_x,
        adaptive[order],
        s=10,
        color=COLORS["controller"],
        alpha=0.62,
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
        linewidth=1.25,
        zorder=4,
    )
    ax.scatter(
        [0.0, 1.0],
        [mean_always, mean_adaptive],
        marker="D",
        s=34,
        color=[COLORS["candidate"], COLORS["controller"]],
        edgecolor="white",
        linewidth=0.45,
        zorder=5,
    )
    ax.text(
        0.02,
        mean_always + 0.09,
        f"{mean_always:.3f}",
        fontsize=5.0,
        color=COLORS["candidate"],
        ha="left",
    )
    ax.text(
        0.98,
        mean_adaptive - 0.12,
        f"{mean_adaptive:.3f}",
        fontsize=5.0,
        color=COLORS["controller"],
        ha="right",
        va="top",
    )
    ax.text(
        0.50,
        2.28,
        f"{reduction:.1f}% lower\nmean time",
        fontsize=5.4,
        color=COLORS["charcoal"],
        ha="center",
        va="center",
    )
    ax.text(
        0.98,
        0.97,
        "64/64 identical\ncontrols and HF traces",
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
    ax.set_xticklabels(
        [
            r"Always level 1" "\n" r"$704/704$ at $\mathcal{N}_1$",
            r"Adaptive" "\n" r"$294/704$ at $\mathcal{N}_1$",
        ],
        fontsize=5.1,
    )
    ax.set_ylabel("Check time (s per decision)")
    clean_axes(ax, grid=False)
    panel_label(ax, "b", x=-0.30, y=1.08)


def trajectory_heatmap_panel(fig: plt.Figure, spec) -> None:
    labels, raw, verified = load_unsafe_traces()
    frame = fig.add_subplot(spec)
    frame.set_axis_off()
    panel_label(frame, "c", x=-0.075, y=1.02)

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
        "thermal_risk",
        [COLORS["controller"], "#F7F7F7", COLORS["unsafe"]],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=0.96, vcenter=1.0, vmax=1.013)
    image = None
    for ax, matrix, title, color, count in [
        (raw_ax, raw, "Raw surrogate", COLORS["unsafe"], "6/6 cross"),
        (
            verified_ax,
            verified,
            "Verified continuation",
            COLORS["controller"],
            "0/6 cross",
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
        ax.set_title(
            f"{title}   {count}",
            fontsize=5.8,
            color=color,
            fontweight="bold",
            pad=3,
        )
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
    verified_ax.set_yticklabels([])
    raw_ax.set_ylabel("Held-out episode")
    assert image is not None
    colorbar = fig.colorbar(image, cax=cbar_ax, extend="min")
    colorbar.set_ticks([0.96, 0.98, 1.00, 1.01])
    colorbar.ax.tick_params(labelsize=4.8, width=0.45, length=2.0)
    colorbar.outline.set_linewidth(0.45)
    colorbar.ax.set_title("HF risk", fontsize=5.0, pad=3)
    colorbar.ax.axhline(
        norm(1.0) * (colorbar.ax.get_ylim()[1] - colorbar.ax.get_ylim()[0])
        + colorbar.ax.get_ylim()[0],
        color=COLORS["charcoal"],
        linewidth=0.5,
    )


def safety_task_panel(ax: plt.Axes) -> None:
    data = load_safety_task_frontier()
    raw = data["raw surrogate"]
    verified = data["verified continuation"]
    continuation = data["K=5 continuation only"]

    raw_xy = (raw["unsafe_rate"], raw["task"])
    verified_xy = (verified["unsafe_rate"], verified["task"])
    continuation_xy = (continuation["unsafe_rate"], continuation["task"])

    ax.add_patch(
        FancyArrowPatch(
            raw_xy,
            verified_xy,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=COLORS["charcoal"],
            connectionstyle="arc3,rad=0.14",
            zorder=2,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.10, continuation_xy[1] + 0.25),
            (0.10, verified_xy[1] - 0.25),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=COLORS["controller"],
            zorder=2,
        )
    )

    ax.scatter(
        *raw_xy,
        marker="s",
        s=56,
        color=COLORS["unsafe"],
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    ax.scatter(
        *verified_xy,
        marker="D",
        s=65,
        color=COLORS["controller"],
        edgecolor="white",
        linewidth=0.55,
        zorder=5,
    )
    ax.scatter(
        *continuation_xy,
        marker="o",
        s=54,
        color=COLORS["certificate"],
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )

    ax.annotate(
        "Raw surrogate\n6/128 unsafe; 100% task",
        raw_xy,
        xytext=(-5, -2),
        textcoords="offset points",
        fontsize=5.4,
        color=COLORS["unsafe"],
        ha="right",
        va="top",
    )
    ax.text(
        0.18,
        100.86,
        "Verified continuation\n0/128 unsafe; 99.95% task",
        fontsize=5.6,
        color=COLORS["controller"],
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        r"$K=5$ continuation only"
        "\n0/128 unsafe; 91.73% task",
        continuation_xy,
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=5.3,
        color=COLORS["certificate"],
        ha="left",
        va="center",
    )
    ax.text(
        2.65,
        100.62,
        r"$6\rightarrow0$ unsafe",
        fontsize=5.5,
        color=COLORS["charcoal"],
        ha="center",
        va="bottom",
    )
    ax.text(
        0.35,
        95.75,
        "+8.22 percentage\npoints of task",
        fontsize=5.2,
        color=COLORS["controller"],
        ha="left",
        va="center",
    )
    ax.set_title(
        "Held-out safety-task outcome",
        loc="left",
        fontsize=6.2,
        fontweight="bold",
        color=COLORS["charcoal"],
        pad=7,
    )
    ax.text(
        0.98,
        1.012,
        r"$n=128$",
        transform=ax.transAxes,
        fontsize=5.2,
        color=COLORS["unavailable"],
        ha="right",
        va="bottom",
    )
    ax.text(
        0.35,
        99.25,
        "95% upper confidence limit: 2.31%",
        fontsize=4.9,
        color=COLORS["unavailable"],
        ha="left",
        va="top",
    )

    ax.set_xlim(-0.45, 5.55)
    ax.set_ylim(90.7, 101.35)
    ax.set_xticks([0, 2, 4])
    ax.set_yticks([92, 96, 100])
    ax.set_xlabel(r"Unsafe episodes (%)  $\leftarrow$ lower")
    ax.set_ylabel("Mean HF task (% of raw)")
    ax.yaxis.set_label_coords(-0.065, 0.5)
    clean_axes(ax, grid=False)
    panel_label(ax, "d", x=-0.15, y=1.025)


def main() -> None:
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 5.20))
    outer = fig.add_gridspec(
        2,
        12,
        height_ratios=[0.88, 1.12],
        hspace=0.43,
        wspace=0.78,
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.955,
    )
    decision_chain_panel(fig.add_subplot(outer[0, :4]))
    validation_cost_panel(fig.add_subplot(outer[0, 4:7]))
    trajectory_heatmap_panel(fig, outer[1, :7])
    safety_task_panel(fig.add_subplot(outer[:, 7:]))
    save_figure(fig, OUT_STEM.with_suffix(".pdf"))
    print(OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

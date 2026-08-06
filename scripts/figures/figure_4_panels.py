"""Mismatch diagnosis and recoverability panels for Figure 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figure_4_data_adapter as data_adapter
from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    MAX_HEIGHT_IN,
    PDE_COLORS,
    clean_axes,
    panel_label,
)


OUT = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "Figure_4_panels.pdf"


def panel_a_unresolved_outcomes(ax: plt.Axes) -> None:
    rows = data_adapter.base.failure_rows()
    x = np.arange(len(rows), dtype=float)
    unsafe = np.asarray([row["unsafe_execution"] for row in rows], dtype=float)
    unavailable = np.asarray(
        [row["continuation_unavailable"] for row in rows], dtype=float
    )
    total = unsafe + unavailable

    unsafe_fill = "#E8A7AA"
    unavailable_fill = "#B9BEC8"
    width = 0.42
    ax.bar(
        x, unsafe, width=width, color=unsafe_fill, edgecolor=COLORS["unsafe"],
        linewidth=0.55, zorder=3,
    )
    ax.bar(
        x, unavailable, bottom=unsafe, width=width, color=unavailable_fill,
        edgecolor=COLORS["unavailable"], linewidth=0.55, zorder=2,
    )
    ax.plot(
        x, total, color=COLORS["charcoal"], marker="D", markersize=4.2,
        linewidth=1.0, zorder=4,
    )

    for xpos, n_unsafe, n_unavailable, n_total in zip(
        x, unsafe, unavailable, total
    ):
        if n_unsafe > 0:
            ax.text(
                xpos, n_unsafe / 2, f"{int(n_unsafe)}", ha="center",
                va="center", fontsize=5.4, color=COLORS["charcoal"],
                fontweight="bold",
            )
        else:
            ax.text(
                xpos - 0.15, 1.3, "0", ha="center", va="center",
                fontsize=5.1, color=COLORS["unsafe"], fontweight="bold",
            )
        ax.text(
            xpos, n_unsafe + n_unavailable / 2, f"{int(n_unavailable)}",
            ha="center", va="center", fontsize=5.4,
            color=COLORS["charcoal"], fontweight="bold",
        )
        ax.text(
            xpos, n_total + 2.0, f"{int(n_total)}", ha="center", va="bottom",
            fontsize=5.7, color=COLORS["charcoal"], fontweight="bold",
        )

    handles = [
        Patch(facecolor=unsafe_fill, edgecolor=COLORS["unsafe"],
              label="Unsafe execution"),
        Patch(facecolor=unavailable_fill, edgecolor=COLORS["unavailable"],
              label="Unavailable verified continuation"),
        Line2D([], [], marker="D", linestyle="-", color=COLORS["charcoal"],
               markersize=3.5, label="Unresolved outcome"),
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.995),
        ncols=2, fontsize=4.65, handlelength=1.15, handletextpad=0.35,
        columnspacing=0.72, labelspacing=0.30, borderaxespad=0.0,
    )

    ax.set_xticks(
        x,
        [
            "Direct surrogate\nselection",
            "Residual-based\nmismatch detector",
            "HF reference\nlabels",
        ],
    )
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(0, 66)
    ax.set_ylabel("Outcomes among 720 scenarios")
    ax.set_title(
        "Decision validity and physical recoverability remain distinct",
        loc="left", fontsize=6.25, fontweight="bold", pad=3.0,
    )
    clean_axes(ax, grid=True)
    panel_label(ax, "a", x=-0.13, y=1.03)


def panel_b_hf_margin(ax: plt.Axes) -> None:
    rows = data_adapter.base.continuation_margin_rows()
    y_lookup = {"burgers": 2.0, "grayscott": 1.0, "kolmogorov": 0.0}
    rng = np.random.default_rng(20260716)

    ax.axvspan(-7.0, 0.0, color=COLORS["light_red"], alpha=0.72, linewidth=0)
    ax.axvspan(0.0, 18.0, color=COLORS["light_green"], alpha=0.20, linewidth=0)
    for pde in data_adapter.base.PDE_ORDER:
        group = [row for row in rows if row["pde"] == pde]
        x = np.asarray([row["normalized_hf_margin_percent"] for row in group])
        y = y_lookup[pde] + rng.uniform(-0.16, 0.16, size=len(group))
        ax.scatter(
            x, y, s=10, color=PDE_COLORS[pde], alpha=0.28,
            edgecolors="none", zorder=2,
        )

    unsafe = [
        row for row in rows
        if row["exact_label_floor_role"] == "unsafe_continuation"
    ]
    rejected = [
        row for row in rows
        if row["exact_label_floor_role"] == "safe_certificate_abstention"
    ]
    ax.scatter(
        [row["normalized_hf_margin_percent"] for row in unsafe],
        [y_lookup[row["pde"]] for row in unsafe],
        s=29, marker="D", color=COLORS["unsafe"], edgecolor="white",
        linewidth=0.45, zorder=5,
    )
    ax.scatter(
        [row["normalized_hf_margin_percent"] for row in rejected],
        [y_lookup[row["pde"]] for row in rejected],
        s=44, marker="o", facecolor="white", edgecolor=COLORS["certificate"],
        linewidth=1.3, zorder=6,
    )

    ax.axvline(
        0.0, color=COLORS["charcoal"], linewidth=0.75,
        linestyle=(0, (3, 2)), zorder=3,
    )
    ax.text(
        -0.18, 2.40, "unsafe", ha="right", fontsize=5.1,
        color=COLORS["unsafe"],
    )
    ax.text(
        0.18, 2.40, "safe", ha="left", fontsize=5.1,
        color=COLORS["controller"],
    )

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=PDE_COLORS["burgers"],
               markersize=3.3, label="Burgers"),
        Line2D([], [], marker="o", linestyle="none", color=PDE_COLORS["grayscott"],
               markersize=3.3, label="Gray-Scott"),
        Line2D([], [], marker="o", linestyle="none", color=PDE_COLORS["kolmogorov"],
               markersize=3.3, label="Kolmogorov"),
        Line2D([], [], marker="D", linestyle="none", color=COLORS["unsafe"],
               markeredgecolor="white", markeredgewidth=0.35, markersize=4.0,
               label="HF-unsafe"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=COLORS["certificate"], markeredgewidth=1.0,
               markersize=4.2, label="Certificate rejection"),
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.985),
        ncols=3, fontsize=4.6, handletextpad=0.28, columnspacing=0.72,
        labelspacing=0.34, borderaxespad=0.0,
    )

    ax.set_yticks(
        [2, 1, 0],
        [data_adapter.base.PDE_LABELS[pde] for pde in data_adapter.base.PDE_ORDER],
    )
    ax.set_xlim(-7, 18)
    ax.set_ylim(-0.45, 3.02)
    ax.set_xlabel("Normalized HF safety margin (%)")
    ax.set_title(
        "HF replay distinguishes unsafe continuations from certificate rejection",
        loc="left", fontsize=6.25, fontweight="bold", pad=3.0,
    )
    clean_axes(ax, grid=False)
    panel_label(ax, "b", x=-0.055, y=1.03)


def panel_c_recoverability(ax: plt.Axes) -> None:
    rows = data_adapter.base.action_stress_rows()
    column_labels = (
        "K5", "K15", "HF-CEM", "1.25x authority", "2x authority",
        "4x authority",
    )
    by_key = {
        (row["pde"], row["ic_idx"], row["short_label"]): row for row in rows
    }

    ordered_keys: list[tuple[str, int]] = []
    for pde in ("burgers", "grayscott"):
        keys = sorted({(row["pde"], row["ic_idx"]) for row in rows
                       if row["pde"] == pde})

        def first_recovered(key: tuple[str, int]) -> tuple[int, int]:
            outcomes = [
                int(by_key[(key[0], key[1], label)]["hf_safe"])
                for label in column_labels
            ]
            first = next((idx for idx, value in enumerate(outcomes) if value), 99)
            return first, key[1]

        ordered_keys.extend(sorted(keys, key=first_recovered))

    matrix = np.asarray(
        [
            [int(by_key[(pde, ic_idx, label)]["hf_safe"])
             for label in column_labels]
            for pde, ic_idx in ordered_keys
        ],
        dtype=int,
    )
    assert matrix.shape == (10, 6)
    assert list(np.sum(matrix, axis=0)) == [0, 0, 0, 1, 3, 4]

    ax.axvspan(-0.5, 2.5, color=COLORS["light_gray"], alpha=0.65, linewidth=0)
    ax.axvspan(2.5, 5.5, color=COLORS["light_green"], alpha=0.36, linewidth=0)
    for row_index in range(matrix.shape[0]):
        ax.hlines(
            row_index, -0.35, 5.35, color="white", linewidth=0.55,
            alpha=0.95, zorder=1,
        )
        for column_index in range(matrix.shape[1]):
            if matrix[row_index, column_index]:
                ax.scatter(
                    column_index, row_index, s=25, marker="o",
                    facecolor=COLORS["controller"], edgecolor="white",
                    linewidth=0.55, zorder=4,
                )
            else:
                ax.scatter(
                    column_index, row_index, s=17, marker="x",
                    color=COLORS["unsafe"], linewidth=0.8, alpha=0.82,
                    zorder=3,
                )

    safe_counts = np.sum(matrix, axis=0)
    for column_index, count in enumerate(safe_counts):
        color = COLORS["controller"] if count else COLORS["charcoal"]
        ax.text(
            column_index, -0.96, f"{int(count)}/10", ha="center", va="bottom",
            fontsize=5.6, fontweight="bold", color=color,
        )

    ax.axvline(
        2.5, color=COLORS["charcoal"], linewidth=0.75,
        linestyle=(0, (3, 2)),
    )
    ax.axhline(2.5, color="white", linewidth=1.1, zorder=2)
    ax.text(
        1.0, -1.72, "Search under original actuator bounds", ha="center",
        fontsize=5.45, color=COLORS["charcoal"], fontweight="bold",
    )
    ax.text(
        4.0, -1.72, "Expanded actuator bounds", ha="center",
        fontsize=5.45, color=COLORS["controller"], fontweight="bold",
    )

    ax.set_xticks(
        np.arange(6),
        ["K=5", "K=15", "HF-CEM", "+25%", "+100%", "+300%"],
    )
    ax.set_yticks(
        [1.0, 6.0],
        ["Burgers\n(3 states)", "Gray-Scott\n(7 states)"],
    )
    ax.get_yticklabels()[0].set_color(PDE_COLORS["burgers"])
    ax.get_yticklabels()[1].set_color(PDE_COLORS["grayscott"])
    ax.tick_params(axis="y", length=0, pad=4.0)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(9.5, -2.75)
    ax.set_title(
        "The admissible control set defines the observed recoverability boundary",
        loc="left", fontsize=6.45, fontweight="bold", pad=3.0,
    )
    clean_axes(ax, grid=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "c", x=-0.055, y=1.07)


def panel_d_sequence_evaluation(fig: plt.Figure, spec) -> None:
    raw_rows = data_adapter.base.ablation_rows()
    rows = [
        {
            **raw_rows[0], "label": "Full-horizon sequence",
            "color": COLORS["candidate"], "marker": "o",
        },
        {
            **raw_rows[1], "label": "Continuation-augmented sequence",
            "color": COLORS["controller"], "marker": "D",
        },
        {
            **raw_rows[2], "label": "Executed sequence",
            "color": COLORS["unavailable"], "marker": "s",
        },
    ]
    y = np.asarray([2.0, 1.0, 0.0])

    frame = fig.add_subplot(spec)
    frame.set_axis_off()
    panel_label(frame, "d", x=-0.045, y=1.02)
    frame.text(
        0.0, 1.025,
        "Complete-sequence evaluation preserves a verified continuation",
        transform=frame.transAxes, ha="left", va="bottom", fontsize=6.45,
        fontweight="bold", color=COLORS["charcoal"],
    )

    inner = spec.subgridspec(1, 2, width_ratios=[0.79, 0.21], wspace=0.035)
    ax_rate = fig.add_subplot(inner[0])
    ax_status = fig.add_subplot(inner[1], sharey=ax_rate)

    for ax in (ax_rate, ax_status):
        ax.axhspan(0.55, 1.45, color=COLORS["light_green"], alpha=0.62,
                   linewidth=0)

    for row, ypos in zip(rows, y):
        rate = 100.0 * float(row["execution_rate"])
        ax_rate.text(
            75.7, ypos, row["label"], ha="left", va="center",
            fontsize=5.35, color=COLORS["charcoal"],
        )
        ax_rate.hlines(
            ypos, 79.1, rate, color=COLORS["mid_gray"], linewidth=1.35,
            zorder=2,
        )
        ax_rate.scatter(
            rate, ypos, s=35, marker=row["marker"], color=row["color"],
            edgecolor="white", linewidth=0.55, zorder=4,
        )
        ax_rate.text(
            rate + 0.35, ypos, f"{row['candidate_releases']}/96",
            ha="left", va="center", fontsize=5.25, color=row["color"],
            fontweight="bold",
        )

        retained = bool(row["continuation_available_next_update"])
        if retained:
            ax_status.scatter(
                0.28, ypos, s=33, marker="o", color=COLORS["controller"],
                edgecolor="white", linewidth=0.55, zorder=4,
            )
            status = "retained"
            status_color = COLORS["controller"]
        else:
            ax_status.scatter(
                0.28, ypos, s=38, marker="x", color=COLORS["unsafe"],
                linewidth=1.05, zorder=4,
            )
            status = "unavailable"
            status_color = COLORS["unsafe"]
        ax_status.text(
            0.46, ypos, status, ha="left", va="center", fontsize=5.2,
            color=status_color,
        )

    ax_rate.set_xlim(75.5, 89.5)
    ax_rate.set_ylim(-0.55, 2.72)
    ax_rate.set_xticks([80, 84, 88])
    ax_rate.set_yticks([])
    ax_rate.set_xlabel("Optimized-block execution (%)")
    clean_axes(ax_rate, grid=False)
    ax_rate.spines["left"].set_visible(False)
    ax_rate.spines["bottom"].set_bounds(80, 88)

    ax_status.set_xlim(0.0, 1.0)
    ax_status.set_ylim(-0.55, 2.72)
    ax_status.set_axis_off()
    ax_status.text(
        0.28, 2.50, "Verified continuation\nat next update",
        ha="center", va="bottom", fontsize=5.05, linespacing=0.95,
        color=COLORS["charcoal"],
    )


def validate_data() -> None:
    failure = data_adapter.base.failure_rows()
    assert [row["unsafe_execution"] for row in failure] == [17, 0, 0]
    assert [row["continuation_unavailable"] for row in failure] == [33, 48, 44]
    assert [row["overall_failure"] for row in failure] == [50, 48, 44]

    margins = data_adapter.base.continuation_margin_rows()
    assert len(margins) == 180
    assert sum(row["exact_label_floor_role"] == "unsafe_continuation"
               for row in margins) == 10
    rejected = [row for row in margins
                if row["exact_label_floor_role"] == "safe_certificate_abstention"]
    assert len(rejected) == 1 and rejected[0]["pde"] == "grayscott"

    ablation = data_adapter.base.ablation_rows()
    assert [row["candidate_releases"] for row in ablation] == [77, 82, 82]
    assert [row["unsafe_episodes"] for row in ablation] == [0, 0, 0]


def main() -> None:
    validate_data()

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, min(4.82, MAX_HEIGHT_IN)))
    outer = fig.add_gridspec(
        3, 1, height_ratios=[1.00, 0.93, 0.52], hspace=0.53,
        left=0.082, right=0.992, bottom=0.070, top=0.969,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.30)

    panel_a_unresolved_outcomes(fig.add_subplot(top[0]))
    panel_b_hf_margin(fig.add_subplot(top[1]))
    panel_c_recoverability(fig.add_subplot(outer[1]))
    panel_d_sequence_evaluation(fig, outer[2])
    data_adapter.save_outputs(fig, OUT)

    print(OUT)
    print("panel a: unsafe [17,0,0], unavailable [33,48,44]")
    print("panel b: 60 states per PDE; 10 HF-unsafe; 1 certificate rejection")
    print("panel c: recovered [0,0,0,1,3,4]")
    print("panel d: executions [77,82,82]/96; continuation [yes,yes,no]")


if __name__ == "__main__":
    main()

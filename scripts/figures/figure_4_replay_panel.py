"""High-fidelity replay panel and shared outcome encodings for Figure 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import figure_4_data_adapter as data_adapter
import figure_4_panels as panels
from plot_style import (
    COLORS as BASE_COLORS,
    DOUBLE_COLUMN_IN,
    MAX_HEIGHT_IN,
    clean_axes,
    panel_label,
)


OUT = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "Figure_4_replay_panel.pdf"

COLORS = {
    **BASE_COLORS,
    "candidate": "#3F70B5",
    "controller": "#248879",
    "unsafe": "#8A5AA3",
    "certificate": "#D19A2A",
    "unavailable": "#667CB4",
    "charcoal": "#27364A",
    "mid_gray": "#91A5C3",
    "light_gray": "#EEF3FA",
    "light_blue": "#E8F0FA",
    "light_green": "#E8F4F1",
    "light_gold": "#FBF3DE",
    "light_red": "#F2EDF7",
}

PDE_COLORS = {
    "burgers": "#3F70B5",
    "grayscott": "#7B6DB2",
    "kolmogorov": "#2C9184",
}

# Reuse the shared state-grouping logic with the manuscript palette.
panels.COLORS = COLORS
panels.PDE_COLORS = PDE_COLORS


def panel_a_outcomes(ax: plt.Axes) -> None:
    rows = data_adapter.base.failure_rows()
    x = np.arange(len(rows), dtype=float)
    unsafe = np.asarray([row["unsafe_execution"] for row in rows], dtype=float)
    unavailable = np.asarray(
        [row["continuation_unavailable"] for row in rows], dtype=float
    )
    total = unsafe + unavailable

    unsafe_fill = "#C9B4D9"
    unavailable_fill = "#B8CBE6"
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

    ax.axvspan(-7.0, 0.0, color=COLORS["light_red"], alpha=0.90, linewidth=0)
    ax.axvspan(0.0, 18.0, color=COLORS["light_green"], alpha=0.55, linewidth=0)
    for pde in data_adapter.base.PDE_ORDER:
        group = [row for row in rows if row["pde"] == pde]
        x = np.asarray([row["normalized_hf_margin_percent"] for row in group])
        y = y_lookup[pde] + rng.uniform(-0.16, 0.16, size=len(group))
        ax.scatter(
            x, y, s=10, color=PDE_COLORS[pde], alpha=0.30,
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
        Line2D([], [], marker="D", linestyle="none", color=COLORS["unsafe"],
               markeredgecolor="white", markeredgewidth=0.35, markersize=4.0,
               label="HF-unsafe"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=COLORS["certificate"], markeredgewidth=1.0,
               markersize=4.2, label="Certificate rejection"),
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.985),
        ncols=2, fontsize=4.7, handletextpad=0.30, columnspacing=0.80,
        borderaxespad=0.0,
    )
    ax.set_yticks(
        [2, 1, 0],
        [data_adapter.base.PDE_LABELS[pde] for pde in data_adapter.base.PDE_ORDER],
    )
    for tick, pde in zip(ax.get_yticklabels(), data_adapter.base.PDE_ORDER):
        tick.set_color(PDE_COLORS[pde])
    ax.set_xlim(-7, 18)
    ax.set_ylim(-0.45, 2.93)
    ax.set_xlabel("Normalized HF safety margin (%)")
    ax.set_title(
        "HF replay distinguishes unsafe continuations from certificate rejection",
        loc="left", fontsize=6.25, fontweight="bold", pad=3.0,
    )
    clean_axes(ax, grid=False)
    panel_label(ax, "b", x=-0.055, y=1.03)


def panel_d_compact(ax: plt.Axes) -> None:
    raw_rows = data_adapter.base.ablation_rows()
    rows = [
        {
            **raw_rows[0], "label": "Full-horizon\nsequence",
            "color": COLORS["candidate"], "marker": "o",
        },
        {
            **raw_rows[1], "label": "Continuation-\naugmented sequence",
            "color": COLORS["controller"], "marker": "D",
        },
        {
            **raw_rows[2], "label": "Executed\nsequence",
            "color": COLORS["unavailable"], "marker": "s",
        },
    ]
    y_positions = [0.72, 0.45, 0.18]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    panel_label(ax, "d", x=-0.075, y=1.04)
    ax.text(
        0.0, 1.045,
        "Complete-sequence evaluation\npreserves a verified continuation",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6.1,
        fontweight="bold", color=COLORS["charcoal"], linespacing=1.02,
    )

    ax.add_patch(
        Rectangle(
            (0.0, 0.335), 1.0, 0.23, transform=ax.transAxes,
            facecolor=COLORS["light_green"], edgecolor="none", alpha=0.85,
        )
    )
    ax.text(0.60, 0.92, "Block\nexecution", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.05, linespacing=0.95)
    ax.text(0.855, 0.92, "Next-update\ncontinuation", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.05, linespacing=0.95)

    for row, ypos in zip(rows, y_positions):
        ax.text(
            0.02, ypos, row["label"], transform=ax.transAxes, ha="left",
            va="center", fontsize=5.05, color=COLORS["charcoal"],
            linespacing=0.96,
        )
        ax.scatter(
            0.535, ypos, s=31, marker=row["marker"], color=row["color"],
            edgecolor="white", linewidth=0.5, transform=ax.transAxes, zorder=4,
        )
        ax.text(
            0.595, ypos, f"{row['candidate_releases']}/96",
            transform=ax.transAxes, ha="left", va="center", fontsize=5.2,
            color=row["color"], fontweight="bold",
        )

        retained = bool(row["continuation_available_next_update"])
        if retained:
            ax.scatter(
                0.79, ypos, s=30, marker="o", color=COLORS["controller"],
                edgecolor="white", linewidth=0.5, transform=ax.transAxes,
                zorder=4,
            )
            status = "retained"
            status_color = COLORS["controller"]
        else:
            ax.scatter(
                0.79, ypos, s=36, marker="x", color=COLORS["unsafe"],
                linewidth=1.05, transform=ax.transAxes, zorder=4,
            )
            status = "unavailable"
            status_color = COLORS["unsafe"]
        ax.text(
            0.845, ypos, status, transform=ax.transAxes, ha="left", va="center",
            fontsize=4.75, color=status_color,
        )


def validate_data() -> None:
    failure = data_adapter.base.failure_rows()
    assert [row["unsafe_execution"] for row in failure] == [17, 0, 0]
    assert [row["continuation_unavailable"] for row in failure] == [33, 48, 44]

    margins = data_adapter.base.continuation_margin_rows()
    assert len(margins) == 180
    assert sum(row["exact_label_floor_role"] == "unsafe_continuation"
               for row in margins) == 10
    assert sum(row["exact_label_floor_role"] == "safe_certificate_abstention"
               for row in margins) == 1

    stress = data_adapter.base.action_stress_rows()
    labels = ("K5", "K15", "HF-CEM", "1.25x authority", "2x authority",
              "4x authority")
    assert [sum(row["short_label"] == label and row["hf_safe"] for row in stress)
            for label in labels] == [0, 0, 0, 1, 3, 4]

    ablation = data_adapter.base.ablation_rows()
    assert [row["candidate_releases"] for row in ablation] == [77, 82, 82]


def main() -> None:
    validate_data()

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, min(4.12, MAX_HEIGHT_IN)))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[0.92, 1.08], hspace=0.40,
        left=0.082, right=0.992, bottom=0.075, top=0.963,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.30)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.42, 0.82], wspace=0.22)

    panel_a_outcomes(fig.add_subplot(top[0]))
    panel_b_hf_margin(fig.add_subplot(top[1]))
    panels.panel_c_recoverability(fig.add_subplot(bottom[0]))
    panel_d_compact(fig.add_subplot(bottom[1]))
    data_adapter.save_outputs(fig, OUT)

    print(OUT)
    print("palette: blue / teal / violet / gold")
    print("layout: panels c and d share the lower row")


if __name__ == "__main__":
    main()

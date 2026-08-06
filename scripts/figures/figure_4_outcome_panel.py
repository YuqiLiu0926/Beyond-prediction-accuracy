"""Decision outcomes and compact sequence-comparison panels for Figure 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import figure_4_panels as panels
import figure_4_replay_panel as replay_panel
from plot_style import (
    COLORS as BASE_COLORS,
    DOUBLE_COLUMN_IN,
    MAX_HEIGHT_IN,
    clean_axes,
    panel_label,
)


OUT = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "Figure_4_outcomes.pdf"

COLORS = {
    **replay_panel.COLORS,
    "unsafe": BASE_COLORS["unsafe"],
    "light_red": BASE_COLORS["light_red"],
}

PDE_COLORS = {
    "burgers": "#3F70B5",
    "grayscott": BASE_COLORS["unsafe"],
    "kolmogorov": "#2C9184",
}

# Reused panels look up these palettes at draw time.
replay_panel.COLORS = COLORS
replay_panel.PDE_COLORS = PDE_COLORS
panels.COLORS = COLORS
panels.PDE_COLORS = PDE_COLORS


def panel_a_outcomes(ax: plt.Axes) -> None:
    rows = replay_panel.data_adapter.base.failure_rows()
    x = np.arange(len(rows), dtype=float)
    unsafe = np.asarray([row["unsafe_execution"] for row in rows], dtype=float)
    unavailable = np.asarray(
        [row["continuation_unavailable"] for row in rows], dtype=float
    )
    total = unsafe + unavailable

    unsafe_fill = "#E8A7AA"
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


def panel_d_centered(ax: plt.Axes) -> None:
    raw_rows = replay_panel.data_adapter.base.ablation_rows()
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
    y_positions = [0.51, 0.35, 0.19]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    panel_label(ax, "d", x=-0.045, y=0.88)
    ax.text(
        0.50, 0.88,
        "Complete-sequence evaluation\npreserves a verified continuation",
        transform=ax.transAxes, ha="center", va="center", fontsize=6.1,
        fontweight="bold", color=COLORS["charcoal"], linespacing=1.02,
    )

    ax.add_patch(
        Rectangle(
            (0.01, 0.275), 0.98, 0.15, transform=ax.transAxes,
            facecolor=COLORS["light_green"], edgecolor="none", alpha=0.85,
        )
    )
    ax.text(0.59, 0.68, "Block\nexecution", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.05, linespacing=0.95)
    ax.text(0.85, 0.68, "Next-update\ncontinuation", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.05, linespacing=0.95)

    for row, ypos in zip(rows, y_positions):
        ax.text(
            0.03, ypos, row["label"], transform=ax.transAxes, ha="left",
            va="center", fontsize=5.05, color=COLORS["charcoal"],
            linespacing=0.96,
        )
        ax.scatter(
            0.525, ypos, s=31, marker=row["marker"], color=row["color"],
            edgecolor="white", linewidth=0.5, transform=ax.transAxes, zorder=4,
        )
        ax.text(
            0.585, ypos, f"{row['candidate_releases']}/96",
            transform=ax.transAxes, ha="left", va="center", fontsize=5.2,
            color=row["color"], fontweight="bold",
        )

        retained = bool(row["continuation_available_next_update"])
        if retained:
            ax.scatter(
                0.785, ypos, s=30, marker="o", color=COLORS["controller"],
                edgecolor="white", linewidth=0.5, transform=ax.transAxes,
                zorder=4,
            )
            status = "retained"
            status_color = COLORS["controller"]
        else:
            ax.scatter(
                0.785, ypos, s=36, marker="x", color=COLORS["unsafe"],
                linewidth=1.05, transform=ax.transAxes, zorder=4,
            )
            status = "unavailable"
            status_color = COLORS["unsafe"]
        ax.text(
            0.84, ypos, status, transform=ax.transAxes, ha="left", va="center",
            fontsize=4.75, color=status_color,
        )


def main() -> None:
    replay_panel.validate_data()

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, min(4.12, MAX_HEIGHT_IN)))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[0.92, 1.08], hspace=0.40,
        left=0.082, right=0.992, bottom=0.075, top=0.963,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.30)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.42, 0.82], wspace=0.08)

    panel_a_outcomes(fig.add_subplot(top[0]))
    replay_panel.panel_b_hf_margin(fig.add_subplot(top[1]))
    panels.panel_c_recoverability(fig.add_subplot(bottom[0]))
    panel_d_centered(fig.add_subplot(bottom[1]))
    replay_panel.data_adapter.save_outputs(fig, OUT)

    print(OUT)
    print("palette: original red restored for unsafe outcomes and Gray-Scott")
    print("layout: panel d centred in the lower-right region; c-d gap reduced")


if __name__ == "__main__":
    main()

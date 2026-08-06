"""Complete-sequence evaluation panel for Figure 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figure_4_panels as panels
import figure_4_replay_panel as replay_panel
import figure_4_outcome_panel as outcome_panel
from plot_style import (
    DOUBLE_COLUMN_IN,
    MAX_HEIGHT_IN,
    clean_axes,
    panel_label,
)


OUT = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "Figure_4_sequence_panel.pdf"
COLORS = outcome_panel.COLORS
PDE_COLORS = outcome_panel.PDE_COLORS

# Keep reused panels on the manuscript red, blue and teal palette.
replay_panel.COLORS = COLORS
replay_panel.PDE_COLORS = PDE_COLORS
panels.COLORS = COLORS
panels.PDE_COLORS = PDE_COLORS


def panel_d_bars(frame: plt.Axes) -> None:
    raw_rows = replay_panel.data_adapter.base.ablation_rows()
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
    baseline = 78.0

    frame.set_axis_off()
    panel_label(frame, "d", x=-0.045, y=0.89)
    frame.text(
        0.50, 0.88,
        "Complete-sequence evaluation\npreserves future feasibility",
        transform=frame.transAxes, ha="center", va="center", fontsize=6.1,
        fontweight="bold", color=COLORS["charcoal"], linespacing=1.02,
    )

    # The two result columns form one compact block centred in the panel.
    ax_rate = frame.inset_axes([0.025, 0.15, 0.695, 0.52])
    ax_status = frame.inset_axes([0.725, 0.15, 0.255, 0.52], sharey=ax_rate)

    for ax in (ax_rate, ax_status):
        ax.axhspan(0.55, 1.45, color=COLORS["light_green"], alpha=0.78,
                   linewidth=0)

    for row, ypos in zip(rows, y):
        rate = 100.0 * float(row["execution_rate"])
        ax_rate.text(
            baseline, ypos + 0.24, row["label"], ha="left", va="bottom",
            fontsize=4.9, color=COLORS["charcoal"],
        )
        ax_rate.barh(
            ypos, rate - baseline, left=baseline, height=0.14,
            color=row["color"], alpha=0.28, edgecolor=row["color"],
            linewidth=0.45, zorder=2,
        )
        ax_rate.scatter(
            rate, ypos, s=31, marker=row["marker"], color=row["color"],
            edgecolor="white", linewidth=0.5, zorder=4,
        )
        ax_rate.text(
            rate + 0.30, ypos, f"{row['candidate_releases']}/96",
            ha="left", va="center", fontsize=4.95, color=row["color"],
            fontweight="bold",
        )

        retained = bool(row["continuation_available_next_update"])
        if retained:
            ax_status.scatter(
                0.22, ypos, s=30, marker="o", color=COLORS["controller"],
                edgecolor="white", linewidth=0.5, zorder=4,
            )
            status = "retained"
            status_color = COLORS["controller"]
        else:
            ax_status.scatter(
                0.22, ypos, s=36, marker="x", color=COLORS["unsafe"],
                linewidth=1.05, zorder=4,
            )
            status = "unavailable"
            status_color = COLORS["unsafe"]
        ax_status.text(
            0.39, ypos, status, ha="left", va="center", fontsize=4.75,
            color=status_color,
        )

    ax_rate.set_xlim(77.8, 88.5)
    ax_rate.set_ylim(-0.38, 2.54)
    ax_rate.set_xticks([80, 84, 88], ["80%", "84%", "88%"])
    ax_rate.set_yticks([])
    ax_rate.set_title(
        "Block execution", fontsize=5.15, fontweight="normal", pad=2.0,
    )
    clean_axes(ax_rate, grid=False)
    ax_rate.spines["left"].set_visible(False)
    ax_rate.spines["bottom"].set_bounds(80, 88)

    ax_status.set_xlim(0.0, 1.0)
    ax_status.set_ylim(-0.38, 2.54)
    ax_status.set_axis_off()
    ax_status.text(
        0.50, 1.055, "Next-update\ncontinuation",
        transform=ax_status.transAxes, ha="center", va="bottom",
        fontsize=5.05, linespacing=0.95, color=COLORS["charcoal"],
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

    ax_a = fig.add_subplot(top[0])
    outcome_panel.panel_a_outcomes(ax_a)
    ax_a.set_title(
        "Decision reliability and physical recoverability impose distinct limits",
        loc="left", fontsize=6.25, fontweight="bold", pad=3.0,
    )

    ax_b = fig.add_subplot(top[1])
    replay_panel.panel_b_hf_margin(ax_b)
    ax_b.set_title(
        "HF replay separates physical infeasibility from certification conservatism",
        loc="left", fontsize=6.25, fontweight="bold", pad=3.0,
    )

    ax_c = fig.add_subplot(bottom[0])
    panels.panel_c_recoverability(ax_c)
    ax_c.set_title(
        "The available control set defines the observed recoverability boundary",
        loc="left", fontsize=6.25, fontweight="bold", pad=3.0,
    )

    panel_d_bars(fig.add_subplot(bottom[1]))
    replay_panel.data_adapter.save_outputs(fig, OUT)

    print(OUT)
    print("titles: revised for panels a-d")
    print("panel d: compact block-execution bars with row labels above")


if __name__ == "__main__":
    main()

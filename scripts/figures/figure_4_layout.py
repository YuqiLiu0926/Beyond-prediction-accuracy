"""Compose the publication layout for Figure 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import figure_4_panels as panels
import figure_4_replay_panel as replay_panel
import figure_4_outcome_panel as outcome_panel
import figure_4_sequence_panel as sequence_panel
from plot_style import DOUBLE_COLUMN_IN, MAX_HEIGHT_IN


OUT = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "Figure_4.pdf"


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
    legend = ax_b.get_legend()
    legend.set_loc("upper right")
    legend.set_bbox_to_anchor((0.995, 0.985))

    ax_c = fig.add_subplot(bottom[0])
    panels.panel_c_recoverability(ax_c)
    ax_c.set_title(
        "The available control set defines the observed recoverability boundary",
        loc="left", fontsize=6.25, fontweight="bold", pad=3.0,
    )

    frame_d = fig.add_subplot(bottom[1])
    sequence_panel.panel_d_bars(frame_d)
    ax_rate = frame_d.child_axes[0]
    for label in ax_rate.texts:
        x_pos, y_pos = label.get_position()
        if label.get_text() == "82/96" and abs(y_pos - 1.0) < 1e-9:
            label.set_x(x_pos + 0.25)

    replay_panel.data_adapter.save_outputs(fig, OUT)
    print(OUT)
    print("panel b: legend moved to upper right")
    print("panel d: continuation-augmented 82/96 shifted right")


if __name__ == "__main__":
    main()

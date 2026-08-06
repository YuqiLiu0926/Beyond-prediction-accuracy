"""Generate Figure 4 on decision reliability and physical recoverability."""

from pathlib import Path

import figure_4_outcome_panel as outcome_panel
import figure_4_layout as layout


OUT = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "Figure_4.pdf"
_panel_a = outcome_panel.panel_a_outcomes


def panel_a_oracle_labels(ax):
    _panel_a(ax)
    ax.set_xticks(
        [0, 1, 2],
        [
            "Direct surrogate\nselection",
            "Residual-based\nmismatch detector",
            "HF oracle\nlabels",
        ],
    )


def main() -> None:
    outcome_panel.panel_a_outcomes = panel_a_oracle_labels
    layout.OUT = OUT
    layout.main()


if __name__ == "__main__":
    main()

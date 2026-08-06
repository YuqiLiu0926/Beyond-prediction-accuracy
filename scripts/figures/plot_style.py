"""Shared Nature Portfolio figure style for the NMI rewrite."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


MM_TO_IN = 1.0 / 25.4
SINGLE_COLUMN_IN = 89.0 * MM_TO_IN
# NMI's journal-specific AIP guide caps submitted panels at 180 mm, which is
# slightly stricter than the 183 mm double-column width in Nature's general guide.
DOUBLE_COLUMN_IN = 180.0 * MM_TO_IN
MAX_HEIGHT_IN = 170.0 * MM_TO_IN

COLORS = {
    "candidate": "#3B6FB6",
    "controller": "#2A8C7B",
    "unsafe": "#C84E52",
    "certificate": "#D69E2E",
    "unavailable": "#737B8C",
    "charcoal": "#2F343B",
    "mid_gray": "#A3A8B3",
    "light_gray": "#EEF0F3",
    "light_blue": "#E8F0FA",
    "light_green": "#E7F3F0",
    "light_gold": "#FBF3DE",
    "light_red": "#F8E8E9",
}

PDE_COLORS = {
    "burgers": "#3B6FB6",
    "grayscott": "#C84E52",
    "kolmogorov": "#2A8C7B",
    "thermal": "#D69E2E",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7.0,
            "axes.linewidth": 0.65,
            "axes.edgecolor": COLORS["charcoal"],
            "axes.labelcolor": COLORS["charcoal"],
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "xtick.color": COLORS["charcoal"],
            "ytick.color": COLORS["charcoal"],
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 6.0,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "patch.linewidth": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "stixsans",
            "axes.unicode_minus": False,
        }
    )


def panel_label(ax, label: str, x: float = -0.14, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        fontstyle="normal",
        ha="left",
        va="bottom",
        color=COLORS["charcoal"],
        clip_on=False,
    )


def clean_axes(ax, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9DCE2", linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)


def save_figure(fig, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the target 89 mm or 180 mm canvas exactly.  Vector PDF is the
    # manuscript source; the companion files support editorial production and
    # visual QA without changing the plotted data.
    fig.savefig(path)
    if path.suffix.lower() == ".pdf":
        fig.savefig(path.with_suffix(".svg"))
        fig.savefig(path.with_suffix(".png"), dpi=600)
        fig.savefig(
            path.with_suffix(".tiff"),
            dpi=600,
            pil_kwargs={"compression": "tiff_lzw"},
        )
    plt.close(fig)
    return path


apply_style()

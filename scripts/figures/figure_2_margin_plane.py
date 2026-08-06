"""Figure 2: full safety-margin plane for all 12 surrogate--PDE pairs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from plot_style import COLORS, DOUBLE_COLUMN_IN, panel_label, save_figure


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "external_data" / "decision_archives" / "discovery"
OUT = ROOT / "outputs" / "figures" / "Figure_2.pdf"

PDES = ("burgers", "grayscott", "kolmogorov")
MODELS = ("deeponet", "fno", "pino", "pinn")
PDE_LABELS = {"burgers": "Burgers", "grayscott": "Gray--Scott", "kolmogorov": "Kolmogorov"}
MODEL_LABELS = {"deeponet": "DeepONet", "fno": "FNO", "pino": "PINO", "pinn": "PINN"}

REGIME_STYLE = {
    "SA": {"label": "Safe agreement", "color": COLORS["controller"], "marker": "o", "size": 11},
    "FS": {"label": "False safe", "color": COLORS["unsafe"], "marker": "D", "size": 18},
    "CR": {"label": "Conservative rejection", "color": COLORS["candidate"], "marker": "^", "size": 17},
    "UA": {"label": "Unsafe agreement", "color": COLORS["charcoal"], "marker": "x", "size": 14},
}


def classify(margin_sur: np.ndarray, margin_hf: np.ndarray) -> np.ndarray:
    return np.where(
        (margin_sur >= 0) & (margin_hf >= 0),
        "SA",
        np.where(
            (margin_sur >= 0) & (margin_hf < 0),
            "FS",
            np.where((margin_sur < 0) & (margin_hf >= 0), "CR", "UA"),
        ),
    )


def axis_limits(values: np.ndarray) -> tuple[float, float]:
    low = min(float(np.min(values)), 0.0)
    high = max(float(np.max(values)), 0.0)
    span = high - low
    if span <= 0:
        span = 1.0
    return low - 0.055 * span, high + 0.055 * span


def shade_quadrants(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    x0, x1 = xlim
    y0, y1 = ylim
    if x1 > 0 and y1 > 0:
        ax.add_patch(Rectangle((0, 0), x1, y1, facecolor=COLORS["light_green"], edgecolor="none", alpha=0.32, zorder=0))
    if x1 > 0 and y0 < 0:
        ax.add_patch(Rectangle((0, y0), x1, -y0, facecolor=COLORS["light_red"], edgecolor="none", alpha=0.42, zorder=0))
    if x0 < 0 and y1 > 0:
        ax.add_patch(Rectangle((x0, 0), -x0, y1, facecolor=COLORS["light_blue"], edgecolor="none", alpha=0.36, zorder=0))
    if x0 < 0 and y0 < 0:
        ax.add_patch(Rectangle((x0, y0), -x0, -y0, facecolor=COLORS["light_gray"], edgecolor="none", alpha=0.42, zorder=0))


def load_pair(pde: str, model: str) -> dict:
    path = DATA_ROOT / pde / model / "challenge_conditions" / "decision_data.npz"
    data = np.load(path, allow_pickle=True)
    z_sur = np.asarray(data["z_sur"], float)
    z_hf = np.asarray(data["z_hf"], float)
    z_limit = np.asarray(data["z_limit"], float)
    margin_sur = np.asarray(data["margin_sur"], float)
    margin_hf = np.asarray(data["margin_hf"], float)
    assert len(z_sur) == 200
    assert np.max(np.abs(margin_sur - (z_limit - z_sur))) < 5e-6
    assert np.max(np.abs(margin_hf - (z_limit - z_hf))) < 5e-6
    return {"margin_sur": margin_sur, "margin_hf": margin_hf, "regime": classify(margin_sur, margin_hf)}


def main() -> None:
    fig, axes = plt.subplots(3, 4, figsize=(DOUBLE_COLUMN_IN, 6.05))
    pooled = {key: 0 for key in ("SA", "FS", "CR", "UA")}

    for row, pde in enumerate(PDES):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            data = load_pair(pde, model)
            x = data["margin_sur"]
            y = data["margin_hf"]
            regime = data["regime"]
            xlim = axis_limits(x)
            ylim = axis_limits(y)
            shade_quadrants(ax, xlim, ylim)

            overlap_low = max(xlim[0], ylim[0])
            overlap_high = min(xlim[1], ylim[1])
            if overlap_low < overlap_high:
                ax.plot([overlap_low, overlap_high], [overlap_low, overlap_high], color=COLORS["mid_gray"], linestyle="--", linewidth=0.65, zorder=1)
            ax.axhline(0, color=COLORS["charcoal"], linewidth=0.70, zorder=1)
            ax.axvline(0, color=COLORS["charcoal"], linewidth=0.70, zorder=1)

            for key in ("SA", "CR", "UA", "FS"):
                mask = regime == key
                pooled[key] += int(np.sum(mask))
                style = REGIME_STYLE[key]
                if style["marker"] == "x":
                    ax.scatter(x[mask], y[mask], s=style["size"], marker=style["marker"], color=style["color"], linewidths=0.55, alpha=0.62, zorder=2)
                else:
                    ax.scatter(
                        x[mask],
                        y[mask],
                        s=style["size"],
                        marker=style["marker"],
                        facecolor=style["color"],
                        edgecolor="white" if key in ("FS", "CR") else "none",
                        linewidth=0.35,
                        alpha=0.72 if key != "FS" else 0.90,
                        zorder=3 if key == "FS" else 2,
                    )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(labelsize=5.6, direction="out", length=2.3, width=0.55)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 0:
                ax.text(0.5, 1.04, MODEL_LABELS[model], transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0, fontweight="bold")
            if col == 0:
                ax.set_ylabel(PDE_LABELS[pde] + "\n" + r"HF margin $m_{\mathrm{HF}}$", fontsize=6.3)
            if row == 2:
                ax.set_xlabel(r"Surrogate margin $m_{\mathrm{sur}}$", fontsize=6.3)

            n_fs = int(np.sum(regime == "FS"))
            n_cr = int(np.sum(regime == "CR"))
            annotation = f"FS {n_fs}/200"
            if n_cr:
                annotation += f"\nCR {n_cr}/200"
            ax.text(
                0.97,
                0.05,
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.5,
                color=COLORS["charcoal"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
            )
            panel_label(ax, chr(ord("a") + row * 4 + col), x=-0.12, y=1.03)

    assert pooled == {"SA": 1675, "FS": 116, "CR": 117, "UA": 492}
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=REGIME_STYLE[key]["marker"],
            markerfacecolor=REGIME_STYLE[key]["color"] if key != "UA" else "none",
            markeredgecolor=REGIME_STYLE[key]["color"],
            markersize=4.7,
            label=REGIME_STYLE[key]["label"],
        )
        for key in ("SA", "FS", "CR", "UA")
    ]
    fig.legend(handles=handles, loc="lower center", ncols=4, frameon=False, fontsize=6.2, columnspacing=1.1, handletextpad=0.35, bbox_to_anchor=(0.5, 0.006))
    fig.subplots_adjust(left=0.078, right=0.995, top=0.94, bottom=0.095, wspace=0.27, hspace=0.32)
    save_figure(fig, OUT)
    print(f"{OUT}\nPooled regimes: {pooled}")


if __name__ == "__main__":
    main()

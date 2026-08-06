"""Shared data adapters and panel utilities for Figure 4."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

import figure_4_data as base
from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    MAX_HEIGHT_IN,
    PDE_COLORS,
    clean_axes,
    panel_label,
)


OUT = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "Figure_4_data_adapter.pdf"


def save_outputs(fig, output: Path) -> None:
    fig.savefig(output)
    fig.savefig(output.with_suffix(".svg"))
    png_path = output.with_suffix(".png")
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    rgba = iio.imread(png_path)
    rgb = rgba[:, :, :3] if rgba.ndim == 3 and rgba.shape[2] == 4 else rgba
    tifffile.imwrite(
        output.with_suffix(".tiff"),
        rgb,
        photometric="rgb",
        compression="deflate",
        resolution=(600, 600),
        resolutionunit="INCH",
    )


def panel_a_failure_decomposition(ax) -> None:
    rows = base.failure_rows()
    x = np.arange(len(rows), dtype=float)
    unsafe = np.asarray([row["unsafe_execution"] for row in rows], dtype=float)
    unavailable = np.asarray([row["continuation_unavailable"] for row in rows], dtype=float)
    total = unsafe + unavailable

    width = 0.42
    ax.bar(
        x, unsafe, width=width, color=COLORS["unsafe"], edgecolor="white",
        linewidth=0.45, label="Unsafe execution", zorder=3,
    )
    ax.bar(
        x, unavailable, bottom=unsafe, width=width, color=COLORS["unavailable"],
        alpha=0.82, edgecolor="white", linewidth=0.45,
        label="No verified continuation", zorder=2,
    )
    ax.plot(
        x, total, color=COLORS["charcoal"], marker="D", markersize=4.2,
        linewidth=1.0, label="Overall failure", zorder=4,
    )

    for index, (xpos, unsafe_count, unavailable_count, total_count) in enumerate(
        zip(x, unsafe, unavailable, total)
    ):
        if index == 0:
            ax.text(xpos, unsafe_count / 2, "unsafe\n17", ha="center", va="center",
                    fontsize=4.9, color="white", fontweight="bold", linespacing=0.92)
            ax.text(xpos, unsafe_count + unavailable_count / 2,
                    "no verified\ncontinuation\n33", ha="center", va="center",
                    fontsize=4.9, color="white", fontweight="bold", linespacing=0.90)
        else:
            ax.text(
                xpos, unsafe_count + unavailable_count / 2, f"{int(unavailable_count)}",
                ha="center", va="center", fontsize=5.2, color="white",
                fontweight="bold",
            )
        ax.text(
            xpos, total_count + 2.0, f"{int(total_count)}", ha="center",
            va="bottom", fontsize=5.6, color=COLORS["charcoal"],
            fontweight="bold",
        )

    ax.text(0.08, 52.0, "overall failure", ha="left", va="bottom",
            fontsize=5.0, color=COLORS["charcoal"])

    ax.set_xticks(x, ["Raw\nsurrogate", "Residual\ndetector", "HF\nlabels"])
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(0, 64)
    ax.set_ylabel("Intervention failures / 720")
    ax.set_title(
        "Detection leaves an intervention floor",
        loc="left", fontsize=6.35, fontweight="bold", pad=3.0,
    )
    clean_axes(ax, grid=True)
    panel_label(ax, "a", x=-0.13, y=1.03)


def panel_b_continuation_margins(ax) -> None:
    rows = base.continuation_margin_rows()
    y_lookup = {"burgers": 2.0, "grayscott": 1.0, "kolmogorov": 0.0}
    rng = np.random.default_rng(20260716)

    ax.axvspan(-7.0, 0.0, color=COLORS["light_red"], alpha=0.72, linewidth=0)
    ax.axvspan(0.0, 18.0, color=COLORS["light_green"], alpha=0.20, linewidth=0)
    for pde in base.PDE_ORDER:
        group = [row for row in rows if row["pde"] == pde]
        x = np.asarray([row["normalized_hf_margin_percent"] for row in group])
        y = y_lookup[pde] + rng.uniform(-0.16, 0.16, size=len(group))
        ax.scatter(
            x, y, s=10, color=PDE_COLORS[pde], alpha=0.28,
            edgecolors="none", zorder=2,
        )

    unsafe = [row for row in rows if row["exact_label_floor_role"] == "unsafe_continuation"]
    abstain = [
        row for row in rows
        if row["exact_label_floor_role"] == "safe_certificate_abstention"
    ]
    ax.scatter(
        [row["normalized_hf_margin_percent"] for row in unsafe],
        [y_lookup[row["pde"]] for row in unsafe],
        s=28, marker="D", color=COLORS["unsafe"], edgecolor="white",
        linewidth=0.4, zorder=5,
    )
    ax.scatter(
        [row["normalized_hf_margin_percent"] for row in abstain],
        [y_lookup[row["pde"]] for row in abstain],
        s=42, marker="o", facecolor="white", edgecolor=COLORS["certificate"],
        linewidth=1.25, zorder=6,
    )

    ax.axvline(0.0, color=COLORS["charcoal"], linewidth=0.75, linestyle=(0, (3, 2)))
    ax.text(-0.18, 2.38, "unsafe", ha="right", fontsize=5.2, color=COLORS["unsafe"])
    ax.text(0.18, 2.38, "safe", ha="left", fontsize=5.2, color=COLORS["controller"])
    ax.text(
        -5.8, 0.42, "10 states\n40/44 unresolved cases", color=COLORS["unsafe"],
        fontsize=5.25, ha="left", va="bottom",
    )
    ax.annotate(
        "1 state\n4/44 cases",
        xy=(abstain[0]["normalized_hf_margin_percent"], y_lookup["grayscott"]),
        xytext=(3.25, 0.43), fontsize=5.15, color=COLORS["certificate"],
        arrowprops={
            "arrowstyle": "-", "color": COLORS["certificate"],
            "linewidth": 0.55, "shrinkA": 2.0, "shrinkB": 2.0,
        },
    )

    ax.set_yticks([2, 1, 0], [base.PDE_LABELS[pde] for pde in base.PDE_ORDER])
    ax.set_xlim(-7, 18)
    ax.set_ylim(-0.45, 2.55)
    ax.set_xlabel(r"HF margin of selected continuation $(z_{\rm lim}-Z_{\rm HF})/z_{\rm lim}$ (%)")
    ax.set_title(
        "Unsafe continuations set the exact-label floor",
        loc="left", fontsize=6.35, fontweight="bold", pad=3.0,
    )
    clean_axes(ax, grid=False)
    panel_label(ax, "b", x=-0.055, y=1.03)


def panel_c_recoverability_boundary(ax) -> None:
    rows = base.action_stress_rows()
    state_keys: list[tuple[str, int]] = []
    for pde in ("burgers", "grayscott"):
        state_keys.extend(
            sorted({(row["pde"], row["ic_idx"]) for row in rows if row["pde"] == pde})
        )
    labels = ("K5", "K15", "HF-CEM", "1.25x authority", "2x authority", "4x authority")
    by_key = {(row["pde"], row["ic_idx"], row["short_label"]): row for row in rows}
    matrix = np.asarray(
        [
            [int(by_key[(pde, ic_idx, label)]["hf_safe"]) for label in labels]
            for pde, ic_idx in state_keys
        ],
        dtype=int,
    )

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
                    color=COLORS["unsafe"], linewidth=0.8, alpha=0.82, zorder=3,
                )

    safe_counts = np.sum(matrix, axis=0)
    for column_index, count in enumerate(safe_counts):
        color = COLORS["controller"] if count else COLORS["charcoal"]
        ax.text(
            column_index, -0.96, f"{int(count)}/10", ha="center", va="bottom",
            fontsize=5.6, fontweight="bold", color=color,
        )

    ax.axvline(2.5, color=COLORS["charcoal"], linewidth=0.75, linestyle=(0, (3, 2)))
    ax.text(
        1.0, -1.72, "Search within the original control set", ha="center",
        fontsize=5.5, color=COLORS["charcoal"], fontweight="bold",
    )
    ax.text(
        4.0, -1.72, "Expanded actuator authority", ha="center",
        fontsize=5.5, color=COLORS["controller"], fontweight="bold",
    )

    ax.set_xticks(
        np.arange(len(labels)),
        ["K=5", "K=15\nHF selection", "Direct\nHF-CEM", "1.25$\\times$", "2$\\times$", "4$\\times$"],
    )
    y_labels = [f"{'B' if pde == 'burgers' else 'GS'}-{ic_idx}" for pde, ic_idx in state_keys]
    ax.set_yticks(np.arange(len(state_keys)), y_labels)
    for tick, (pde, _) in zip(ax.get_yticklabels(), state_keys):
        tick.set_color(PDE_COLORS[pde])
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(9.5, -2.75)
    ax.set_title(
        "Recoverability changes with actuator authority, not search breadth",
        loc="left", fontsize=6.5, fontweight="bold", pad=3.0,
    )
    clean_axes(ax, grid=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "c", x=-0.055, y=1.07)


def _arrow(
    ax, start: tuple[float, float], end: tuple[float, float], color: str,
    *, connectionstyle: str = "arc3", linewidth: float = 0.8,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=6.5, linewidth=linewidth,
            color=color, connectionstyle=connectionstyle, transform=ax.transAxes,
            clip_on=False,
        )
    )


def _segment(
    ax, x: float, y: float, width: float, height: float, face: str, edge: str,
    text: str, *, fontsize: float = 5.1,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y), width, height, facecolor=face, edgecolor=edge,
            linewidth=0.75, transform=ax.transAxes,
        )
    )
    ax.text(
        x + width / 2, y + height / 2, text, transform=ax.transAxes,
        ha="center", va="center", fontsize=fontsize, color=COLORS["charcoal"],
    )


def panel_d_preservation_rule(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "d", x=-0.045, y=1.02)
    ax.text(
        0.0, 1.01, "Both branches preserve a verified continuation",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3,
        fontweight="bold",
    )

    ax.text(
        0.02, 0.88, "A new complete sequence passes", transform=ax.transAxes,
        fontsize=5.55, fontweight="bold", color=COLORS["controller"],
    )
    _segment(ax, 0.03, 0.65, 0.11, 0.10, COLORS["light_blue"], COLORS["candidate"], r"$V_k$")
    _segment(
        ax, 0.14, 0.65, 0.18, 0.10, COLORS["light_green"], COLORS["controller"],
        r"$T_k^{(j^\star)}$",
    )
    ax.text(
        0.175, 0.59, r"$S_k^{(j^\star)}=V_k\Vert T_k^{(j^\star)}$",
        transform=ax.transAxes, ha="center", fontsize=5.0,
    )
    _arrow(ax, (0.33, 0.70), (0.39, 0.70), COLORS["certificate"])
    _segment(
        ax, 0.39, 0.63, 0.14, 0.14, COLORS["light_gold"], COLORS["certificate"],
        "multilevel\ncheck passes", fontsize=5.0,
    )
    _arrow(ax, (0.54, 0.70), (0.60, 0.70), COLORS["controller"])
    _segment(ax, 0.60, 0.71, 0.17, 0.085, COLORS["light_blue"], COLORS["candidate"], r"apply $V_k$")
    _segment(
        ax, 0.60, 0.57, 0.17, 0.085, COLORS["light_green"], COLORS["controller"],
        r"retain $T_k^{(j^\star)}$", fontsize=4.9,
    )

    ax.text(
        0.02, 0.42, "No new complete sequence passes", transform=ax.transAxes,
        fontsize=5.55, fontweight="bold", color=COLORS["unsafe"],
    )
    _segment(
        ax, 0.03, 0.20, 0.11, 0.10, COLORS["light_green"], COLORS["controller"],
        r"$R_k[0{:}b_k]$", fontsize=4.8,
    )
    _segment(
        ax, 0.14, 0.20, 0.18, 0.10, COLORS["light_green"], COLORS["controller"],
        r"$R_k[b_k{:}h_k]$", fontsize=4.8,
    )
    ax.text(
        0.175, 0.14, r"$R_k\in\mathcal{C}_k$", transform=ax.transAxes,
        ha="center", fontsize=5.0,
    )
    _arrow(
        ax, (0.46, 0.63), (0.34, 0.30), COLORS["unsafe"],
        connectionstyle="arc3,rad=0.18", linewidth=0.7,
    )
    _arrow(ax, (0.33, 0.25), (0.60, 0.25), COLORS["controller"])
    _segment(
        ax, 0.60, 0.27, 0.17, 0.085, COLORS["light_green"], COLORS["controller"],
        "apply retained prefix", fontsize=5.0,
    )
    _segment(
        ax, 0.60, 0.13, 0.17, 0.085, COLORS["light_green"], COLORS["controller"],
        "retain continuation", fontsize=5.0,
    )

    ax.plot(
        [0.80, 0.80], [0.10, 0.86], transform=ax.transAxes,
        color=COLORS["mid_gray"], linewidth=0.6, linestyle=(0, (2, 2)),
    )
    ax.text(
        0.80, 0.89, "next update", transform=ax.transAxes, ha="center",
        fontsize=4.9, color=COLORS["unavailable"],
    )
    _arrow(ax, (0.77, 0.61), (0.83, 0.61), COLORS["controller"])
    _arrow(ax, (0.77, 0.17), (0.83, 0.39), COLORS["controller"], connectionstyle="arc3,rad=-0.12")
    _segment(
        ax, 0.83, 0.34, 0.16, 0.25, COLORS["light_green"], COLORS["controller"],
        "verified at $k+1$\n" + r"$R_{k+1}\in\mathcal{C}_{k+1}$", fontsize=4.9,
    )


def panel_e_checked_object(ax) -> None:
    rows = base.ablation_rows()
    y_positions = np.asarray([2.0, 1.0, 0.0])
    rates = 100.0 * np.asarray([row["execution_rate"] for row in rows])
    colors = [COLORS["candidate"], COLORS["controller"], COLORS["unavailable"]]
    markers = ["o", "D", "s"]

    ax.axhspan(0.55, 1.45, color=COLORS["light_green"], alpha=0.62, linewidth=0)
    for row, ypos, rate, color, marker in zip(rows, y_positions, rates, colors, markers):
        ax.hlines(ypos, 76.0, rate, color=COLORS["mid_gray"], linewidth=1.2, zorder=1)
        ax.scatter(
            rate, ypos, s=34, marker=marker, color=color, edgecolor="white",
            linewidth=0.55, zorder=3,
        )
        ax.text(
            rate + 0.55, ypos, f"{row['candidate_releases']}/96",
            ha="left", va="center", fontsize=5.2, color=color, fontweight="bold",
        )

        retained = bool(row["continuation_available_next_update"])
        if retained:
            ax.scatter(
                97.0, ypos, s=32, marker="o", facecolor=COLORS["controller"],
                edgecolor="white", linewidth=0.5, zorder=3,
            )
            status = "yes"
            status_color = COLORS["controller"]
        else:
            ax.scatter(97.0, ypos, s=34, marker="x", color=COLORS["unsafe"], linewidth=1.0, zorder=3)
            status = "no"
            status_color = COLORS["unsafe"]
        ax.text(98.1, ypos, status, ha="left", va="center", fontsize=5.1, color=status_color)

    ax.axvline(92.0, color=COLORS["mid_gray"], linewidth=0.6, linestyle=(0, (2, 2)))
    ax.text(
        97.0, 2.62, "verified at\nnext update", ha="center", va="bottom",
        fontsize=5.0, linespacing=0.95,
    )
    ax.text(
        76.0, 2.62, "0/16 unsafe episodes for all three checks", ha="left",
        va="bottom", fontsize=5.05, color=COLORS["unavailable"],
    )

    ax.set_yticks(y_positions, [row["label"] for row in rows])
    ax.set_xticks([78, 82, 86, 90])
    ax.set_xlim(76, 101)
    ax.set_ylim(-0.55, 2.82)
    ax.set_xlabel("Surrogate block execution (%)")
    ax.set_title(
        "Complete-sequence checks retain future control",
        loc="left", fontsize=6.2, fontweight="bold", pad=3.0,
    )
    clean_axes(ax, grid=False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(76, 90)
    ax.tick_params(axis="y", length=0, pad=2)
    panel_label(ax, "e", x=-0.09, y=1.02)


def main() -> None:
    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, min(5.35, MAX_HEIGHT_IN)))
    outer = fig.add_gridspec(
        3, 1, height_ratios=[1.00, 0.89, 0.93], hspace=0.48,
        left=0.074, right=0.992, bottom=0.055, top=0.972,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.30)
    bottom = outer[2].subgridspec(1, 2, width_ratios=[1.48, 1.0], wspace=0.38)
    panel_a_failure_decomposition(fig.add_subplot(top[0]))
    panel_b_continuation_margins(fig.add_subplot(top[1]))
    panel_c_recoverability_boundary(fig.add_subplot(outer[1]))
    panel_d_preservation_rule(fig.add_subplot(bottom[0]))
    panel_e_checked_object(fig.add_subplot(bottom[1]))
    save_outputs(fig, OUT)

    print(OUT)
    print("failure totals:", [row["overall_failure"] for row in base.failure_rows()])
    print("continuation-margin states:", len(base.continuation_margin_rows()))
    stress = base.action_stress_rows()
    for label in ("K5", "K15", "HF-CEM", "1.25x authority", "2x authority", "4x authority"):
        print(label, sum(row["short_label"] == label and row["hf_safe"] for row in stress), "/ 10")


if __name__ == "__main__":
    main()

"""Generate Figure 5 from Gray--Scott closed-loop evidence.

The four panels report episode-level safety, paired risk changes, retained
performance and executed-block availability using the frozen evaluation data.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.lines import Line2D
from PIL import Image

import figure_5_data as source
from plot_style import COLORS, DOUBLE_COLUMN_IN, clean_axes, panel_label


FIGURE_DIR = source.FIGURE_DIR
OUT_STEM = FIGURE_DIR / "Figure_5"


def save_bundle(fig: plt.Figure) -> None:
    """Export an editable vector bundle and 600-dpi review files."""
    png_path = OUT_STEM.with_suffix(".png")
    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".svg"))
    fig.savefig(png_path, dpi=600)
    with Image.open(png_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    tifffile.imwrite(
        OUT_STEM.with_suffix(".tiff"),
        rgb,
        photometric="rgb",
        compression="deflate",
        resolution=(600, 600),
        resolutionunit="INCH",
    )
    plt.close(fig)


def _move_panel_label(ax: plt.Axes, label: str, x: float) -> None:
    """Reposition a panel letter created by an imported plotting function."""
    for artist in ax.texts:
        if artist.get_text() == label:
            artist.set_position((x, 1.045))
            artist.set_fontsize(8.0)
            return
    panel_label(ax, label, x=x, y=1.045)


def trace_panel(ax: plt.Axes, data: dict) -> None:
    """Draw aligned temporal high-fidelity risk traces and their legend."""
    source.trace_panel(ax, data)
    for artist in list(ax.texts):
        text = artist.get_text()
        if text == "safety limit":
            artist.remove()
        elif text == "36/48 crossed":
            artist.set_position((0.98, 0.14))
            artist.set_ha("right")
            artist.set_va("bottom")
            artist.set_fontsize(5.6)
        elif text == "0/48 crossed":
            artist.set_position((0.98, 0.075))
            artist.set_ha("right")
            artist.set_va("bottom")
            artist.set_fontsize(5.6)
    legend = ax.get_legend()
    if legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        labels = [
            "Verified continuation" if label == "Certified controller" else label
            for label in labels
        ]
        legend.remove()
        ax.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.008, 0.012),
            ncol=2,
            columnspacing=0.9,
            handlelength=1.5,
            borderaxespad=0,
        )
    _move_panel_label(ax, "a", -0.105)


def paired_peak_panel(ax: plt.Axes, data: dict) -> None:
    """Draw the paired peak-risk comparison in a compact panel."""
    source.paired_peak_panel(ax, data)
    for artist in list(ax.texts):
        if artist.get_text() == "safety limit":
            artist.remove()
    ax.set_xlim(-0.20, 1.20)
    ax.set_xticks([0, 1], ["Raw", "Verified\ncontinuation"])
    ax.tick_params(axis="x", pad=3.0)
    _move_panel_label(ax, "b", -0.205)


def safety_task_frontier_panel(ax: plt.Axes, data: dict) -> None:
    """Draw the safety-performance plane with restrained annotations."""
    definitions = (
        ("$K=5$ continuation only", "continuation_only", COLORS["unavailable"], "s"),
        ("Verified continuation", "continuation_preserving", COLORS["controller"], "D"),
        ("Raw surrogate", "raw", COLORS["unsafe"], "o"),
    )

    for seed in data["seeds"]:
        local = data["seed_summaries"][seed]["by_mode"]
        raw_task = float(local["raw"]["mean_J_hf_episode"])
        points = []
        for _, mode, _, _ in definitions:
            values = local[mode]
            points.append(
                (
                    100.0 * float(values["unsafe_episodes"]) / float(values["episodes"]),
                    100.0 * float(values["mean_J_hf_episode"]) / raw_task,
                )
            )
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=COLORS["light_gray"],
            linewidth=0.65,
            zorder=1,
        )
        for point, (_, _, color, marker) in zip(points, definitions):
            ax.scatter(
                *point,
                s=17,
                marker=marker,
                facecolor="white",
                edgecolor=color,
                linewidth=0.6,
                alpha=0.72,
                zorder=2,
            )

    pooled = data["summary"]["by_mode"]
    pooled_raw_task = float(pooled["raw"]["mean_J_hf_episode"])
    pooled_points = {}
    for label, mode, color, marker in definitions:
        values = pooled[mode]
        point = (
            100.0 * float(values["unsafe_episodes"]) / float(values["episodes"]),
            100.0 * float(values["mean_J_hf_episode"]) / pooled_raw_task,
        )
        pooled_points[mode] = point
        ax.scatter(
            *point,
            s=47,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
    ax.plot(
        [pooled_points[mode][0] for _, mode, _, _ in definitions],
        [pooled_points[mode][1] for _, mode, _, _ in definitions],
        color=COLORS["charcoal"],
        linewidth=0.95,
        zorder=3,
    )

    verified_retention = pooled_points["continuation_preserving"][1]
    continuation_retention = pooled_points["continuation_only"][1]
    assert math.isclose(verified_retention, 24.318612893114874, rel_tol=0, abs_tol=1e-10)
    assert math.isclose(continuation_retention, 2.5674832278208645, rel_tol=0, abs_tol=1e-10)
    assert pooled_points["raw"][0] == 75.0

    ax.text(
        84.0,
        88.0,
        "36/48 unsafe",
        ha="right",
        va="center",
        fontsize=5.3,
        color=COLORS["unsafe"],
    )
    ax.text(
        4.5,
        75.0,
        "0/48 unsafe\n24.3% retained",
        ha="left",
        va="top",
        fontsize=5.3,
        color=COLORS["controller"],
    )
    ax.annotate(
        "0/48 unsafe\n2.6% retained",
        pooled_points["continuation_only"],
        xytext=(8, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=5.3,
        color=COLORS["unavailable"],
    )
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            label=label,
        )
        for label, _, color, marker in definitions
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=3,
        columnspacing=0.72,
        handletextpad=0.30,
        borderaxespad=0.1,
    )
    ax.set_xlim(-5, 86)
    ax.set_ylim(-6, 109)
    ax.set_xticks([0, 25, 50, 75])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Unsafe HF episodes (%)")
    ax.set_ylabel("HF control performance retained (%)")
    clean_axes(ax, grid=False)
    panel_label(ax, "c", x=-0.105, y=1.045)


def continuation_timing_panel(ax: plt.Axes, data: dict) -> None:
    """Draw continuation activations with a concise summary annotation."""
    rows = [
        [str(decision["source"]) for decision in pair["certified"]["decisions"]]
        for pair in data["pairs"]
    ]
    assert len(rows) == 48 and all(len(row) == 6 for row in rows)
    counts = np.asarray(
        [sum(row[update] == "retained_continuation" for row in rows) for update in range(6)],
        dtype=int,
    )
    assert counts.tolist() == [37, 0, 2, 0, 0, 0]
    assert int(counts.sum()) == 39
    x = np.arange(1, 7)

    ax.axhline(0, color=COLORS["mid_gray"], linewidth=0.65, zorder=0)
    ax.vlines(x, 0, counts, color=COLORS["controller"], linewidth=1.25, zorder=1)
    positive = counts > 0
    ax.scatter(
        x[positive],
        counts[positive],
        s=29,
        facecolor=COLORS["controller"],
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax.scatter(
        x[~positive],
        counts[~positive],
        s=18,
        facecolor="white",
        edgecolor=COLORS["mid_gray"],
        linewidth=0.6,
        zorder=2,
    )
    for xpos, count in zip(x, counts):
        ax.text(
            xpos,
            float(count) + (1.55 if count else 1.05),
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=5.4,
            color=COLORS["controller"] if count else COLORS["unavailable"],
        )

    ax.text(
        0.98,
        0.94,
        "249/288 optimized blocks executed\n37/39 activations at update 1",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.25,
        color=COLORS["charcoal"],
    )
    ax.set_xlim(0.62, 6.38)
    ax.set_ylim(-2.0, 42.5)
    ax.set_xticks(x)
    ax.set_yticks([0, 20, 40])
    ax.set_xlabel("Control update")
    ax.set_ylabel("Continuation activations")
    clean_axes(ax, grid=False)
    panel_label(ax, "d", x=-0.205, y=1.045)


def main() -> None:
    data = source.prepare_data()
    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 4.60))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.88, 1.0],
        height_ratios=[1.08, 0.78],
        hspace=0.47,
        wspace=0.25,
        left=0.082,
        right=0.985,
        bottom=0.115,
        top=0.955,
    )
    trace_panel(fig.add_subplot(grid[0, 0]), data)
    paired_peak_panel(fig.add_subplot(grid[0, 1]), data)
    safety_task_frontier_panel(fig.add_subplot(grid[1, 0]), data)
    continuation_timing_panel(fig.add_subplot(grid[1, 1]), data)
    save_bundle(fig)
    print(OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

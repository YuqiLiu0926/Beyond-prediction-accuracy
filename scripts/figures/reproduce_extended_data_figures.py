"""Generate the revised Extended Data figures from the submitted Source Data.

The script changes only visual organization. It reads the authoritative Excel
workbooks, checks the manuscript totals, and exports PDF, SVG and PNG versions
for Extended Data Figs. 1-6.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    PDE_COLORS,
    clean_axes,
    panel_label,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PACKAGE_ROOT / "external_data" / "source_data"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs" / "extended_data"

MODEL_ORDER = ["deeponet", "fno", "pino", "pinn"]
MODEL_LABELS = {
    "deeponet": "DeepONet",
    "fno": "FNO",
    "pino": "PINO",
    "pinn": "PINN",
}
MODEL_MARKERS = {"deeponet": "o", "fno": "s", "pino": "^", "pinn": "D"}
MODEL_COLORS = {
    "deeponet": PDE_COLORS["burgers"],
    "fno": COLORS["certificate"],
    "pino": COLORS["controller"],
    "pinn": COLORS["unavailable"],
}
PDE_ORDER = ["burgers", "grayscott", "kolmogorov"]
PDE_LABELS = {
    "burgers": "Burgers",
    "grayscott": "Gray-Scott",
    "kolmogorov": "Kolmogorov",
}


def read_sheet(fig_no: int, sheet: str) -> pd.DataFrame:
    path = SOURCE_DIR / f"Source_Data_Extended_Data_Fig_{fig_no}.xlsx"
    return pd.read_excel(path, sheet_name=sheet)


def export_figure(fig: plt.Figure, output_dir: Path, fig_no: int, suffix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"Extended_Data_Fig_{fig_no}{suffix}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def add_panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", pad=5, fontweight="bold")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return centre - half, centre + half


def figure_1(output_dir: Path, suffix: str) -> dict:
    data = read_sheet(1, "Panels_a_b")
    assert len(data) == 12
    assert float(data["passive_l2"].max()) < 1.01e-3
    assert data["false_safe_rate"].between(0, 100).all()
    assert int(round((2.0 * data["false_safe_rate"]).sum())) == 116

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 3.0))
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.19, top=0.94)

    for pde in PDE_ORDER:
        for model in MODEL_ORDER:
            row = data[(data["pde"] == pde) & (data["model"] == model)]
            if row.empty:
                continue
            ax.scatter(
                row["passive_l2"],
                row["false_safe_rate"],
                s=46,
                marker=MODEL_MARKERS[model],
                color=PDE_COLORS[pde],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

    ax.axvline(1e-3, color=COLORS["mid_gray"], linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(1e-3, 24.9, r"$10^{-3}$", ha="right", va="top", color=COLORS["unavailable"])
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1.55e-3)
    ax.set_ylim(-0.8, 25.5)
    ax.set_xlabel(r"One-step relative $L^2$ error")
    ax.set_ylabel("False-safe rate (%)")
    clean_axes(ax)

    pde_handles = [
        Line2D([], [], marker="o", linestyle="none", color=PDE_COLORS[p], label=PDE_LABELS[p])
        for p in PDE_ORDER
    ]
    model_handles = [
        Line2D(
            [],
            [],
            marker=MODEL_MARKERS[m],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["charcoal"],
            label=MODEL_LABELS[m],
        )
        for m in MODEL_ORDER
    ]
    first = ax.legend(handles=pde_handles, loc="upper left", ncols=1, borderaxespad=0.3)
    ax.add_artist(first)
    ax.legend(handles=model_handles, loc="upper center", ncols=4, borderaxespad=0.3)

    export_figure(fig, output_dir, 1, suffix)
    return {"points": len(data), "panels": 1}


def figure_2(output_dir: Path, suffix: str) -> dict:
    thresholds = read_sheet(2, "Panels_a_b_thresholds")
    optimizers = read_sheet(2, "Panels_c_d_optimizers")
    assert set(thresholds["n_controls"]) == {2400}
    assert int(optimizers["false_safe"].sum()) == 114
    thresholds["threshold"] = thresholds["threshold"].replace(
        {"original_cached": "prespecified"}
    )

    threshold_order = ["prespecified", "pair_global_z0_q75", "pair_global_z0_q90"]
    threshold_labels = {
        "prespecified": "Prespecified",
        "pair_global_z0_q75": r"$q_{75}(Z_0)$",
        "pair_global_z0_q90": r"$q_{90}(Z_0)$",
    }
    thresholds = thresholds.set_index("threshold").loc[threshold_order].reset_index()

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 4.45))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.875,
        hspace=0.48,
        wspace=0.42,
        width_ratios=[1.08, 1.0],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    regime_cols = ["safe_agreement", "conservative_rejection", "unsafe_agreement", "false_safe"]
    regime_labels = ["Safe agreement", "Conservative rejection", "Unsafe agreement", "False safety"]
    regime_colors = [COLORS["controller"], COLORS["certificate"], COLORS["unavailable"], COLORS["unsafe"]]
    y = np.arange(len(thresholds))[::-1]
    left = np.zeros(len(thresholds))
    for column, color in zip(regime_cols, regime_colors):
        values = 100.0 * thresholds[column].to_numpy() / thresholds["n_controls"].to_numpy()
        ax_a.barh(y, values, left=left, height=0.48, color=color, edgecolor="none")
        for ypos, start, value in zip(y, left, values):
            if value >= 12:
                text_color = "white" if color in (COLORS["controller"], COLORS["unavailable"]) else COLORS["charcoal"]
                ax_a.text(start + value / 2.0, ypos, f"{value:.0f}%", ha="center", va="center", color=text_color)
        left += values
    for ypos, count in zip(y, thresholds["false_safe"]):
        ax_a.text(101.1, ypos, f"FS {int(count)}", ha="left", va="center", color=COLORS["unsafe"], fontweight="bold")
    ax_a.set_yticks(y, [threshold_labels[name] for name in threshold_order])
    ax_a.set_xlim(0, 109)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_xlabel("Decision regime (%)")
    ax_a.tick_params(axis="y", length=0)
    clean_axes(ax_a)
    add_panel_title(ax_a, "Decision regimes")
    fig.legend(
        handles=[Patch(facecolor=c, label=l) for c, l in zip(regime_colors, regime_labels)],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncols=4,
        frameon=False,
        borderaxespad=0,
        columnspacing=1.25,
        handlelength=1.2,
    )
    panel_label(ax_a, "a", x=-0.20)

    curve_order = ["pair_global_z0_q75", "pair_global_z0_q90", "prespecified"]
    curve = thresholds.set_index("threshold").loc[curve_order]
    x = 100.0 * curve["surrogate_acceptance_rate"].to_numpy()
    yb = 100.0 * curve["false_safe_fraction_among_accepted"].to_numpy()
    ax_b.plot(x, yb, color=COLORS["mid_gray"], linewidth=1.0, zorder=1)
    marker_map = {"pair_global_z0_q75": "s", "pair_global_z0_q90": "D", "prespecified": "o"}
    color_map = {
        "pair_global_z0_q75": COLORS["certificate"],
        "pair_global_z0_q90": COLORS["certificate"],
        "prespecified": COLORS["candidate"],
    }
    offsets = {
        "pair_global_z0_q75": (1.7, 0.22),
        "pair_global_z0_q90": (1.7, -0.40),
        "prespecified": (-19.0, 0.28),
    }
    short = {"pair_global_z0_q75": r"$q_{75}$", "pair_global_z0_q90": r"$q_{90}$", "prespecified": "Prespecified"}
    for name, xv, yv in zip(curve_order, x, yb):
        ax_b.scatter(xv, yv, marker=marker_map[name], s=48, color=color_map[name], edgecolor="white", linewidth=0.6, zorder=3)
        dx, dy = offsets[name]
        ax_b.text(xv + dx, yv + dy, short[name], ha="left", va="center")
    ax_b.set_xlim(14, 81)
    ax_b.set_ylim(1.55, 7.25)
    ax_b.set_xlabel("Surrogate acceptance (%)")
    ax_b.set_ylabel("False-safe rate (%)")
    clean_axes(ax_b)
    add_panel_title(ax_b, "Acceptance and false safety")
    panel_label(ax_b, "b", x=-0.20)

    method_order = ["cem", "random_shooting"]
    method_labels = {"cem": "CEM", "random_shooting": "Random shooting"}
    method_colors = {"cem": COLORS["candidate"], "random_shooting": COLORS["certificate"]}
    method_markers = {"cem": "o", "random_shooting": "D"}
    yc = np.arange(len(PDE_ORDER))[::-1]
    for ypos, pde in zip(yc, PDE_ORDER):
        local = optimizers[optimizers["pde"] == pde].set_index("optimizer")
        values = [100.0 * float(local.loc[m, "false_safe_rate"]) for m in method_order]
        ax_c.plot(values, [ypos, ypos], color=COLORS["mid_gray"], linewidth=0.9, zorder=1)
        for j, method in enumerate(method_order):
            count = int(local.loc[method, "false_safe"])
            total = int(local.loc[method, "n_controls"])
            ax_c.scatter(values[j], ypos, marker=method_markers[method], s=46, color=method_colors[method], edgecolor="white", linewidth=0.5, zorder=3)
            offset_y = 0.14 if method == "cem" else -0.20
            ax_c.text(values[j], ypos + offset_y, f"{count}/{total}", ha="center", va="center", color=method_colors[method], fontsize=5.6)
    ax_c.set_yticks(yc, [PDE_LABELS[p] for p in PDE_ORDER])
    ax_c.set_xlim(-0.2, 6.55)
    ax_c.set_ylim(-0.24, 2.28)
    ax_c.set_xlabel("False-safe rate (%)")
    ax_c.tick_params(axis="y", length=0)
    clean_axes(ax_c)
    add_panel_title(ax_c, "Across PDEs")
    ax_c.legend(
        handles=[Line2D([], [], marker=method_markers[m], linestyle="none", color=method_colors[m], label=method_labels[m]) for m in method_order],
        loc="lower right",
        ncols=2,
        borderaxespad=0.2,
    )
    panel_label(ax_c, "c", x=-0.20)

    pooled = optimizers.groupby("optimizer", as_index=False).agg(false_safe=("false_safe", "sum"), n_controls=("n_controls", "sum"))
    yd = [1, 0]
    for ypos, method in zip(yd, method_order):
        row = pooled[pooled["optimizer"] == method].iloc[0]
        count, total = int(row["false_safe"]), int(row["n_controls"])
        rate = 100.0 * count / total
        lo, hi = wilson_interval(count, total)
        lo, hi = 100.0 * lo, 100.0 * hi
        ax_d.errorbar(rate, ypos, xerr=[[rate - lo], [hi - rate]], fmt=method_markers[method], color=method_colors[method], ecolor=method_colors[method], markersize=5.0, capsize=2.3, linewidth=1.1)
        ax_d.text(hi + 0.08, ypos, f"{count}/{total:,}", ha="left", va="center")
    ax_d.set_yticks(yd, [method_labels[m] for m in method_order])
    ax_d.set_xlim(0.65, 3.75)
    ax_d.set_xlabel("Pooled false-safe rate (%)")
    ax_d.tick_params(axis="y", length=0)
    clean_axes(ax_d)
    add_panel_title(ax_d, "Pooled estimates")
    panel_label(ax_d, "d", x=-0.20)

    export_figure(fig, output_dir, 2, suffix)
    return {"threshold_candidates": 2400, "optimizer_candidates_per_method": 2400, "panels": 4}


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def figure_3(output_dir: Path, suffix: str) -> dict:
    examples = read_sheet(3, "Panel_a_examples")
    paths = read_sheet(3, "Panels_b_c_paths")
    pair_summary = read_sheet(3, "Panel_c_pair_summary")
    matched = paths[paths["replay_matched"] == 1]
    assert int((matched["regime"] == "FS").sum()) == 115
    assert int((matched["regime"] == "SA").sum()) == 1625

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 4.55))
    grid = fig.add_gridspec(2, 2, left=0.08, right=0.985, bottom=0.105, top=0.96, hspace=0.50, wspace=0.34, height_ratios=[0.92, 1.05])
    top = grid[0, :].subgridspec(1, 2, wspace=0.20)
    ax_a1 = fig.add_subplot(top[0, 0])
    ax_a2 = fig.add_subplot(top[0, 1], sharey=ax_a1)
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    for ax, outcome, title in (
        (ax_a1, "false_safe", "False-safe path"),
        (ax_a2, "safe_agreement", "Safe-agreement path"),
    ):
        local = examples[(examples["example_outcome"] == outcome) & (examples["selected_state"] == 1)].sort_values("progress")
        ax.axhspan(-0.18, 0, color=COLORS["light_red"], zorder=0)
        ax.axhspan(0, 0.135, color=COLORS["light_green"], alpha=0.72, zorder=0)
        ax.axhline(0, color=COLORS["charcoal"], linewidth=0.75)
        ax.plot(local["progress"], local["m_sur"], color=COLORS["candidate"], marker="o", markersize=3.7, label=r"$m_{\mathrm{sur}}$")
        ax.plot(local["progress"], local["m_hf"], color=COLORS["unsafe"], marker="o", markersize=3.7, label=r"$m_{\mathrm{HF}}$")
        start = local.iloc[0]
        final = local.iloc[-1]
        for value, color in ((start["m_sur"], COLORS["candidate"]), (start["m_hf"], COLORS["unsafe"])):
            ax.scatter(start["progress"], value, s=46, facecolor="white", edgecolor=color, linewidth=1.0, zorder=4)
        for value, color in ((final["m_sur"], COLORS["candidate"]), (final["m_hf"], COLORS["unsafe"])):
            ax.scatter(final["progress"], value, marker="*", s=72, color=color, edgecolor=COLORS["charcoal"], linewidth=0.45, zorder=5)
        ax.text(0.96, float(final["m_sur"]) + 0.012, r"$m_{\mathrm{sur}}$", color=COLORS["candidate"], ha="right", va="bottom")
        ax.text(0.96, float(final["m_hf"]) + (0.012 if float(final["m_hf"]) >= 0 else -0.012), r"$m_{\mathrm{HF}}$", color=COLORS["unsafe"], ha="right", va="bottom" if float(final["m_hf"]) >= 0 else "top")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.18, 0.135)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel("Normalized optimization progress")
        add_panel_title(ax, title)
        clean_axes(ax)
    ax_a1.set_ylabel("Safety margin")
    ax_a2.tick_params(labelleft=False)
    panel_label(ax_a1, "a", x=-0.15)

    for regime, color, label in (
        ("FS", COLORS["unsafe"], "False safety"),
        ("SA", COLORS["controller"], "Safe agreement"),
    ):
        values = matched.loc[matched["regime"] == regime, "delta_gap_normalized"].dropna().to_numpy()
        xb, yb = ecdf(values)
        ax_b.step(xb, yb, where="post", color=color, linewidth=1.5, label=f"{label} ($n={len(values):,}$)")
    ax_b.axvspan(0, 0.15, color=COLORS["light_red"], alpha=0.55, zorder=0)
    ax_b.axvline(0, color=COLORS["charcoal"], linestyle=(0, (2, 2)), linewidth=0.8)
    ax_b.set_xlim(-0.04, 0.15)
    ax_b.set_ylim(0, 1.0)
    ax_b.set_xlabel("Normalized signed-risk-gap change")
    ax_b.set_ylabel("Cumulative fraction")
    add_panel_title(ax_b, "Change in signed risk gap")
    ax_b.legend(loc="lower right")
    clean_axes(ax_b)
    panel_label(ax_b, "b", x=-0.16)

    model_order = ["deeponet", "fno", "pino", "pinn"]
    values = np.full((len(PDE_ORDER), len(model_order)), np.nan)
    counts = np.zeros_like(values, dtype=int)
    for yi, pde in enumerate(PDE_ORDER):
        for xi, model in enumerate(model_order):
            row = pair_summary[(pair_summary["pde"] == pde) & (pair_summary["model"] == model)]
            if row.empty:
                continue
            counts[yi, xi] = int(row.iloc[0]["n_false_safe"])
            values[yi, xi] = float(row.iloc[0]["gap_amplified_fraction_false_safe"])
    display_values = values.copy()
    display_values[counts < 3] = np.nan
    cmap = LinearSegmentedColormap.from_list("gap_increase", ["#F2F3F5", "#F7D7D8", COLORS["unsafe"]])
    cmap.set_bad(COLORS["light_gray"])
    image = ax_c.imshow(display_values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for yi in range(values.shape[0]):
        for xi in range(values.shape[1]):
            n = counts[yi, xi]
            value = values[yi, xi]
            if n < 3 or np.isnan(value):
                text = f"$n={n}$"
                color = COLORS["unavailable"]
            else:
                text = f"{100.0 * value:.0f}%\n($n={n}$)"
                color = "white" if value >= 0.68 else COLORS["charcoal"]
            ax_c.text(xi, yi, text, ha="center", va="center", color=color, fontsize=5.7)
    ax_c.set_xticks(range(len(model_order)), [MODEL_LABELS[m] for m in model_order])
    ax_c.set_yticks(range(len(PDE_ORDER)), [PDE_LABELS[p] for p in PDE_ORDER])
    ax_c.tick_params(length=0)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    add_panel_title(ax_c, "Pair-specific gap increase")
    panel_label(ax_c, "c", x=-0.16)
    cbar = fig.colorbar(image, ax=ax_c, orientation="horizontal", fraction=0.11, pad=0.20, ticks=[0, 0.5, 1])
    cbar.set_label("Fraction with increased risk gap", labelpad=2)
    cbar.outline.set_linewidth(0.5)

    export_figure(fig, output_dir, 3, suffix)
    return {"matched_paths": int(len(matched)), "false_safe_paths": 115, "safe_agreement_paths": 1625, "panels": 3}


def figure_4(output_dir: Path, suffix: str) -> dict:
    metrics = read_sheet(4, "Panels_a_b_metrics")
    operating = read_sheet(4, "Panel_c_operating")
    assert set(metrics["n"]) == {1791}
    assert set(metrics["n_fs"]) == {116}

    feature_order = ["compact4", "residual19_only", "compact4_residual19", "phi6_residual19"]
    feature_labels = {
        "compact4": "Risk summaries (4)",
        "residual19_only": "Operator defects (19)",
        "compact4_residual19": "Combined (4 + 19)",
        "phi6_residual19": "Expanded (6 + 19)",
    }
    feature_colors = {
        "compact4": COLORS["candidate"],
        "residual19_only": COLORS["certificate"],
        "compact4_residual19": COLORS["controller"],
        "phi6_residual19": COLORS["charcoal"],
    }
    metrics = metrics.set_index("feature_set").loc[feature_order].reset_index()

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 4.3))
    grid = fig.add_gridspec(2, 2, left=0.19, right=0.985, bottom=0.10, top=0.96, hspace=0.48, wspace=0.36, height_ratios=[0.90, 1.05])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    def interval_panel(ax: plt.Axes, value: str, low: str, high: str, xlim: tuple[float, float], ticks: list[float], title: str, label: str, show_y: bool) -> None:
        y = np.arange(len(metrics))[::-1]
        for ypos, row in zip(y, metrics.itertuples(index=False)):
            name = row.feature_set
            v = float(getattr(row, value))
            lo = float(getattr(row, low))
            hi = float(getattr(row, high))
            color = feature_colors[name]
            ax.errorbar(v, ypos, xerr=[[v - lo], [hi - v]], fmt="o", color=color, ecolor=color, markersize=4.8, capsize=2.2, linewidth=1.1, markeredgecolor="white", markeredgewidth=0.5)
            ax.text(hi + 0.012 * (xlim[1] - xlim[0]), ypos, f"{v:.3f}", ha="left", va="center")
        ax.set_xlim(*xlim)
        ax.set_xticks(ticks)
        ax.set_ylim(-0.55, 3.55)
        ax.set_xlabel(title)
        if show_y:
            ax.set_yticks(y, [feature_labels[name] for name in feature_order])
        else:
            ax.set_yticks(y, [])
        ax.tick_params(axis="y", length=0, pad=3)
        clean_axes(ax)
        panel_label(ax, label, x=-0.25 if show_y else -0.15)

    interval_panel(ax_a, "auroc", "auroc_ci95_lo", "auroc_ci95_hi", (0.64, 0.87), [0.65, 0.70, 0.75, 0.80, 0.85], "AUROC", "a", True)
    interval_panel(ax_b, "average_precision", "ap_ci95_lo", "ap_ci95_hi", (0.10, 0.44), [0.1, 0.2, 0.3, 0.4], "Average precision", "b", False)

    bottom = grid[1, :].subgridspec(1, 2, wspace=0.42)
    ax_c1 = fig.add_subplot(bottom[0, 0])
    ax_c2 = fig.add_subplot(bottom[0, 1])
    op = operating.set_index("feature_set")
    before_names = ["compact4", "phi6"]
    after_names = ["compact4_residual19", "phi6_residual19"]
    row_labels = ["Risk summaries (4)", "Risk summaries (6)"]

    def operating_axis(ax: plt.Axes, key: str, xlabel: str, xlim: tuple[float, float], ticks: list[float], show_y: bool) -> None:
        y = np.array([1, 0])
        before = np.array([float(op.loc[name, key]) for name in before_names])
        after = np.array([float(op.loc[name, key]) for name in after_names])
        for ypos, x0, x1 in zip(y, before, after):
            ax.annotate("", xy=(x1, ypos), xytext=(x0, ypos), arrowprops={"arrowstyle": "-|>", "color": COLORS["mid_gray"], "linewidth": 1.0, "mutation_scale": 7})
            ax.scatter(x0, ypos, s=34, color=COLORS["unavailable"], edgecolor="white", linewidth=0.5, zorder=3)
            ax.scatter(x1, ypos, marker="D", s=38, color=COLORS["controller"], edgecolor="white", linewidth=0.5, zorder=4)
            span = xlim[1] - xlim[0]
            ax.text(x0, ypos + 0.17, f"{int(x0)}", ha="center", color=COLORS["unavailable"])
            ax.text(x1, ypos - 0.20, f"{int(x1)}", ha="center", color=COLORS["controller"])
        ax.set_xlim(*xlim)
        ax.set_xticks(ticks)
        ax.set_ylim(-0.45, 1.45)
        ax.set_xlabel(xlabel)
        if show_y:
            ax.set_yticks(y, row_labels)
        else:
            ax.set_yticks(y, [])
        ax.tick_params(axis="y", length=0)
        clean_axes(ax)

    operating_axis(ax_c1, "accepted_unsafe", "Unsafe acceptances", (24, 43), [25, 30, 35, 40], True)
    operating_axis(ax_c2, "fallback", "Rejected decisions", (875, 1060), [900, 950, 1000, 1050], False)
    ax_c1.text(0.01, 0.98, r"Global CQR, $\alpha=0.1$", transform=ax_c1.transAxes, ha="left", va="top")
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=COLORS["unavailable"], label="Risk summaries"),
        Line2D([], [], marker="D", linestyle="none", color=COLORS["controller"], label="+ operator defects"),
    ]
    ax_c2.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.03), ncols=2, borderaxespad=0, columnspacing=0.8)
    panel_label(ax_c1, "c", x=-0.25)

    export_figure(fig, output_dir, 4, suffix)
    return {"accepted_candidates": 1791, "false_safe_candidates": 116, "panels": 3}


def figure_5(output_dir: Path, suffix: str) -> dict:
    surface = read_sheet(5, "Panels_a_b_summary")
    cem = read_sheet(5, "Panel_c_CEM_summary")
    horizons = [25, 50, 75, 100]
    authorities = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 4.55))
    grid = fig.add_gridspec(2, 2, left=0.09, right=0.90, bottom=0.11, top=0.95, hspace=0.50, wspace=0.32, height_ratios=[1.0, 0.82])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    heat_cmap = LinearSegmentedColormap.from_list("recoverability", [COLORS["light_red"], COLORS["light_gray"], "#5E87A6"])

    def heatmap(ax: plt.Axes, cohort: str, title: str):
        local = surface[(surface["cohort"] == cohort) & (surface["family"] == "dense_constant")]
        frac = local.pivot(index="horizon", columns="authority", values="recoverable_fraction").loc[horizons, authorities]
        count = local.pivot(index="horizon", columns="authority", values="recoverable_states").loc[horizons, authorities]
        total = local.pivot(index="horizon", columns="authority", values="n_states").loc[horizons, authorities]
        image = ax.imshow(frac.to_numpy(), vmin=0, vmax=1, cmap=heat_cmap, aspect="auto")
        for yi in range(len(horizons)):
            for xi in range(len(authorities)):
                f = float(frac.iloc[yi, xi])
                color = "white" if f >= 0.72 else COLORS["charcoal"]
                ax.text(xi, yi, f"{int(count.iloc[yi, xi])}/{int(total.iloc[yi, xi])}", ha="center", va="center", color=color)
        ax.set_xticks(range(len(authorities)), [f"{x:g}" for x in authorities])
        ax.set_yticks(range(len(horizons)), [str(x) for x in horizons])
        ax.set_xlabel("Authority factor")
        ax.set_ylabel("Horizon")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_panel_title(ax, title)
        return image

    image = heatmap(ax_a, "mechanism", "Challenging states ($n=10$)")
    heatmap(ax_b, "independent_boundary_stress", "Near-boundary states ($n=12$)")
    panel_label(ax_a, "a", x=-0.18)
    panel_label(ax_b, "b", x=-0.18)
    cbar_ax = fig.add_axes([0.922, 0.59, 0.012, 0.25])
    cbar = fig.colorbar(image, cax=cbar_ax, ticks=[0, 0.5, 1])
    cbar.set_label("Recoverable fraction", labelpad=3)
    cbar.outline.set_linewidth(0.5)

    mechanism = surface[(surface["cohort"] == "mechanism") & (surface["horizon"] == 100)]
    family_styles = {
        "K5": (COLORS["unavailable"], "o", ":", 5.8, "$K=5$"),
        "K15": (COLORS["unsafe"], "s", "--", 5.0, "$K=15$"),
        "dense_constant": (COLORS["candidate"], "D", "-", 4.2, "33 constant controls"),
    }
    for family, (color, marker, linestyle, size, label) in family_styles.items():
        local = mechanism[mechanism["family"] == family].sort_values("authority")
        ax_c.plot(local["authority"], local["recoverable_states"], color=color, marker=marker, markersize=size, linewidth=1.15, linestyle=linestyle, markerfacecolor="white", markeredgewidth=1.0, label=label)
    cem_best = cem.sort_values(["authority", "recoverable_states", "n_blocks"], ascending=[True, False, False]).groupby("authority", as_index=False).first()
    ax_c.plot(cem_best["authority"], cem_best["recoverable_states"], color=COLORS["controller"], marker="^", markersize=5.0, linewidth=1.15, linestyle="--", label="HF-CEM")
    ax_c.set_xlim(0.82, 4.18)
    ax_c.set_ylim(-0.2, 5.2)
    ax_c.set_xticks(authorities, [f"{x:g}" for x in authorities])
    ax_c.set_yticks(range(0, 6))
    ax_c.set_xlabel("Authority factor")
    ax_c.set_ylabel("Recoverable states")
    clean_axes(ax_c, grid=True)
    add_panel_title(ax_c, "Control families at horizon 100")
    ax_c.legend(loc="upper left", ncols=4, borderaxespad=0.2, columnspacing=1.1)
    panel_label(ax_c, "c", x=-0.08)

    export_figure(fig, output_dir, 5, suffix)
    return {"challenging_states": 10, "near_boundary_states": 12, "panels": 3}


def choose_representative(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    candidate_counts = (
        candidates.assign(predicted_safe=lambda frame: frame["risk_sur"] <= frame["risk_limit"])
        .groupby(["model", "state_idx"], as_index=False)["predicted_safe"]
        .sum()
        .rename(columns={"predicted_safe": "n_predicted_safe"})
    )
    eligible = decisions.merge(candidate_counts, on=["model", "state_idx"])
    eligible = eligible[
        (eligible["selected_matches_hf_oracle"] == 0)
        & eligible["normalized_safe_regret"].notna()
        & eligible["selected_field_relative_l2"].notna()
        & (eligible["n_predicted_safe"] >= 4)
    ].copy()
    eligible["score"] = (
        eligible["normalized_safe_regret"].rank(pct=True)
        + (1.0 - eligible["selected_field_relative_l2"].rank(pct=True))
        + (1.0 - eligible["candidate_task_spearman"].rank(pct=True))
    )
    return eligible.sort_values(["score", "normalized_safe_regret"], ascending=False).iloc[0]


def figure_6(output_dir: Path, suffix: str) -> dict:
    candidates = read_sheet(6, "Panel_a_candidates")
    decisions = read_sheet(6, "Panels_b_c_decisions")
    summary = read_sheet(6, "Panel_b_model_summary").set_index("model")
    assert len(decisions) == 160
    assert int(decisions["selected_false_safe"].sum()) == 3

    representative = choose_representative(decisions, candidates)
    local = candidates[(candidates["model"] == representative["model"]) & (candidates["state_idx"] == representative["state_idx"])].copy()
    local["hf_safe"] = local["risk_hf"] <= local["risk_limit"]
    local = local.sort_values("performance_sur", ascending=False).reset_index(drop=True)
    local["surrogate_rank"] = np.arange(1, len(local) + 1)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 3.18))
    grid = fig.add_gridspec(1, 3, left=0.07, right=0.985, bottom=0.18, top=0.90, wspace=0.42, width_ratios=[1.20, 0.92, 1.35])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    scale = max(float(local["performance_hf"].max()), float(local["performance_sur"].max()))
    x = local["surrogate_rank"].to_numpy()
    y_sur = 100.0 * local["performance_sur"].to_numpy() / scale
    y_hf = 100.0 * local["performance_hf"].to_numpy() / scale
    ax_a.plot(x, y_sur, color=COLORS["candidate"], marker="o", markersize=3.2, linewidth=1.2, label="Surrogate")
    ax_a.plot(x, y_hf, color=COLORS["unsafe"], marker="o", markersize=3.2, linewidth=1.2, label="HF")
    unsafe = ~local["hf_safe"].to_numpy()
    if np.any(unsafe):
        ax_a.scatter(x[unsafe], y_hf[unsafe], facecolor="white", edgecolor=COLORS["unsafe"], s=30, linewidth=1.0, zorder=4, label="HF unsafe")
    selected_idx = int(representative["selected_idx"])
    oracle_idx = int(representative["hf_oracle_idx"])
    selected_row = int(local.index[local["candidate_idx"] == selected_idx][0])
    oracle_row = int(local.index[local["candidate_idx"] == oracle_idx][0])
    ax_a.scatter(x[selected_row], y_hf[selected_row], marker="*", s=72, color=COLORS["candidate"], edgecolor="white", linewidth=0.6, zorder=6, label="Selected")
    ax_a.scatter(x[oracle_row], y_hf[oracle_row], marker="D", s=34, color=COLORS["unsafe"], edgecolor="white", linewidth=0.6, zorder=6, label="HF-safe optimum")
    ax_a.set_xlim(0.4, 16.6)
    ax_a.set_xticks([1, 4, 8, 12, 16])
    ax_a.set_xlabel("Surrogate rank")
    ax_a.set_ylabel("Task value (%)")
    add_panel_title(ax_a, "Surrogate and HF rankings differ")
    ax_a.legend(loc="lower left", borderaxespad=0.2)
    clean_axes(ax_a)
    panel_label(ax_a, "a", x=-0.17)

    rng = np.random.default_rng(20260806)
    for xpos, model in enumerate(MODEL_ORDER):
        values = 100.0 * decisions.loc[decisions["model"] == model, "normalized_safe_regret"].dropna().to_numpy(dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax_b.scatter(np.full(len(values), xpos) + jitter, values, s=10, color=MODEL_COLORS[model], alpha=0.28, linewidth=0)
        row = summary.loc[model]
        mean, lo, hi = float(row["mean_regret_percent"]), float(row["ci_low"]), float(row["ci_high"])
        ax_b.errorbar(xpos, mean, yerr=[[mean - lo], [hi - mean]], fmt=MODEL_MARKERS[model], markersize=5.2, color=MODEL_COLORS[model], markeredgecolor="white", markeredgewidth=0.6, capsize=2.2, linewidth=1.2, zorder=5)
    ax_b.axhline(0, color=COLORS["mid_gray"], linewidth=0.8, linestyle="--")
    ax_b.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=25, ha="right")
    ax_b.set_ylabel("Normalized regret (%)")
    add_panel_title(ax_b, "Regret by surrogate model")
    clean_axes(ax_b)
    panel_label(ax_b, "b", x=-0.18)

    safe = decisions[(decisions["selected_false_safe"] == 0) & decisions["normalized_safe_regret"].notna()]
    for model in MODEL_ORDER:
        local_decisions = safe[safe["model"] == model]
        ax_c.scatter(
            100.0 * local_decisions["selected_field_relative_l2"],
            100.0 * local_decisions["normalized_safe_regret"],
            s=22,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            label=MODEL_LABELS[model],
        )
    ax_c.set_xlabel(r"Rollout relative $L^2$ error (%)")
    ax_c.set_ylabel("Normalized regret (%)")
    add_panel_title(ax_c, "Field error and decision regret")
    ax_c.legend(loc="upper right", borderaxespad=0.2)
    clean_axes(ax_c)
    panel_label(ax_c, "c", x=-0.17)

    export_figure(fig, output_dir, 6, suffix)
    return {"grouped_decisions": 160, "safe_selections": int(len(safe)), "false_safe_selections": 3, "panels": 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suffix", default="_reproduced")
    args = parser.parse_args()

    qa = {
        "figure_1": figure_1(args.output_dir, args.suffix),
        "figure_2": figure_2(args.output_dir, args.suffix),
        "figure_3": figure_3(args.output_dir, args.suffix),
        "figure_4": figure_4(args.output_dir, args.suffix),
        "figure_5": figure_5(args.output_dir, args.suffix),
        "figure_6": figure_6(args.output_dir, args.suffix),
        "source": "Source_Data_Extended_Data_Fig_1.xlsx through Source_Data_Extended_Data_Fig_6.xlsx",
        "panel_changes": {
            "figure_1": "Former panel b removed; direct field-error versus false-safety evidence retained.",
            "figure_3": "Trial-position ticks removed from panel a; illustrative and aggregate evidence retained.",
            "figure_4": "Former conceptual panel d removed; quantitative panels retained.",
        },
    }
    qa_path = args.output_dir / f"Extended_Data_Figure_QA{args.suffix}.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(qa_path)


if __name__ == "__main__":
    main()

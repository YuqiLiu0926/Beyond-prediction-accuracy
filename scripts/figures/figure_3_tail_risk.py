"""Figure 3 composite: tail-risk amplitude, timing and pair-level safety."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.lines import Line2D

import figure_3_components as components
from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    MAX_HEIGHT_IN,
    PDE_COLORS,
    clean_axes,
    panel_label,
)


OUT = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "Figure_3.pdf"
N_PAIR_BOOTSTRAP = 2000


def save_composite(fig, output: Path) -> None:
    """Export vector masters and a losslessly compressed 600-dpi TIFF."""
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


def risk_summary(record: dict, functional: str) -> tuple[np.ndarray, np.ndarray]:
    if functional == "mean":
        return np.mean(record["z_sur_ts"], axis=1), np.mean(record["z_hf_ts"], axis=1)
    if functional == "q90":
        return (
            np.percentile(record["z_sur_ts"], 90, axis=1),
            np.percentile(record["z_hf_ts"], 90, axis=1),
        )
    if functional == "max":
        return np.max(record["z_sur_ts"], axis=1), np.max(record["z_hf_ts"], axis=1)
    raise ValueError(f"Unknown risk functional: {functional}")


def original_risk_scatter(
    ax,
    record: dict,
    functional: str,
    title: str,
    letter: str | None = None,
    show_legend: bool = False,
    maximum_labels: bool = False,
) -> float:
    """Reproduce panels a and b from ``fig_tail_risk_final.pdf`` as vectors."""
    z_sur, z_hf = risk_summary(record, functional)
    false_safe = record["false_safe"]
    correlation = float(np.corrcoef(z_sur, z_hf)[0, 1])

    ax.scatter(
        z_sur[~false_safe], z_hf[~false_safe], s=14,
        color="#4A90D9", alpha=0.50, edgecolors="none", zorder=2,
    )
    ax.scatter(
        z_sur[false_safe], z_hf[false_safe], s=24,
        color="#D9534F", marker="D", alpha=0.90,
        edgecolor=COLORS["charcoal"], linewidth=0.40, zorder=3,
        label=f"False-safe (n={int(np.sum(false_safe))})",
    )
    low = min(float(np.min(z_sur)), float(np.min(z_hf)))
    high = max(float(np.max(z_sur)), float(np.max(z_hf)))
    padding = 0.08 * (high - low)
    low -= padding
    high += padding
    ax.plot(
        [low, high], [low, high], color=COLORS["charcoal"],
        linestyle=(0, (4, 3)), linewidth=0.60, alpha=0.50, zorder=1,
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    if maximum_labels:
        ax.set_xlabel(r"$Z_{\mathrm{sur}}$ (max)", labelpad=1.5)
        ax.set_ylabel(r"$Z_{\mathrm{HF}}$ (max)", labelpad=1.5)
    else:
        ax.set_xlabel(r"$Z_{\mathrm{sur}}$", labelpad=1.5)
        ax.set_ylabel(r"$Z_{\mathrm{HF}}$", labelpad=1.5)
    ax.set_title(title + "\n" + rf"$r = {correlation:.3f}$", fontsize=6.4, pad=2.0)
    if letter:
        panel_label(ax, letter, x=-0.23, y=1.08)
    if show_legend:
        ax.legend(
            loc="upper left", fontsize=5.0, markerscale=0.82,
            handletextpad=0.35, borderaxespad=0.30,
        )
    return correlation


def _bootstrap_correlation_interval(
    x: np.ndarray, y: np.ndarray, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(N_PAIR_BOOTSTRAP, len(x)))
    boot = components._rowwise_correlation(x[indices], y[indices])
    return tuple(float(value) for value in np.percentile(boot[np.isfinite(boot)], [2.5, 97.5]))


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def pair_summary_with_intervals(records: list[dict]) -> list[dict]:
    rows = []
    for pair_index, record in enumerate(records):
        z_sur, z_hf = risk_summary(record, "max")
        correlation = float(np.corrcoef(z_sur, z_hf)[0, 1])
        r_low, r_high = _bootstrap_correlation_interval(z_sur, z_hf, 20260715 + pair_index)
        n = len(z_sur)
        false_safe_count = int(np.sum(record["false_safe"]))
        fs_low, fs_high = _wilson_interval(false_safe_count, n)
        rows.append(
            {
                "pde": record["pde"],
                "surrogate": record["model"],
                "maximum_risk_pearson_r": correlation,
                "maximum_risk_pearson_ci95_low": r_low,
                "maximum_risk_pearson_ci95_high": r_high,
                "false_safe_count": false_safe_count,
                "n_controls": n,
                "false_safe_rate": false_safe_count / n,
                "false_safe_rate_ci95_low": fs_low,
                "false_safe_rate_ci95_high": fs_high,
                "correlation_interval": "percentile bootstrap, 2,000 resamples",
                "rate_interval": "Wilson score interval",
            }
        )
    return rows


def all_pair_panel(ax, records: list[dict]) -> None:
    rows = pair_summary_with_intervals(records)
    correlations = np.asarray([row["maximum_risk_pearson_r"] for row in rows])
    rates = 100.0 * np.asarray([row["false_safe_rate"] for row in rows])

    for row in rows:
        x = row["maximum_risk_pearson_r"]
        y = 100.0 * row["false_safe_rate"]
        xerr = np.maximum(
            0.0,
            np.asarray(
                [[x - row["maximum_risk_pearson_ci95_low"]],
                 [row["maximum_risk_pearson_ci95_high"] - x]]
            ),
        )
        yerr = 100.0 * np.maximum(
            0.0,
            np.asarray(
                [[row["false_safe_rate"] - row["false_safe_rate_ci95_low"]],
                 [row["false_safe_rate_ci95_high"] - row["false_safe_rate"]]]
            ),
        )
        color = PDE_COLORS[row["pde"]]
        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color,
            elinewidth=0.55, capsize=1.4, capthick=0.55, alpha=0.38, zorder=1,
        )
        ax.scatter(
            x, y, s=34, marker=components.MODEL_MARKERS[row["surrogate"]],
            facecolor=color, edgecolor="white", linewidth=0.45, zorder=3,
        )

    callouts = {
        ("grayscott", "pinn"): (0.020, 0.9, "Gray-Scott PINN"),
        ("grayscott", "fno"): (-0.190, 1.0, "Gray-Scott FNO"),
        ("kolmogorov", "deeponet"): (-0.235, 1.1, "Kolmogorov DeepONet"),
        ("burgers", "pino"): (0.030, 1.2, "Burgers PINO"),
    }
    for row in rows:
        key = (row["pde"], row["surrogate"])
        if key not in callouts:
            continue
        dx, dy, label = callouts[key]
        x = row["maximum_risk_pearson_r"]
        y = 100.0 * row["false_safe_rate"]
        ax.annotate(
            label, xy=(x, y), xytext=(x + dx, y + dy),
            fontsize=5.2, color=PDE_COLORS[row["pde"]],
            arrowprops={"arrowstyle": "-", "color": PDE_COLORS[row["pde"]],
                        "linewidth": 0.45, "alpha": 0.72, "shrinkA": 1.5, "shrinkB": 2.5},
        )

    ax.axhline(0, color=COLORS["charcoal"], linewidth=0.55)
    ax.set_xlim(-0.43, 1.055)
    ax.set_ylim(-1.4, 28.2)
    ax.set_xlabel(r"Within-pair maximum-risk correlation $r_{\max}$")
    ax.set_ylabel("False-safe rate (%)")
    ax.set_title("High maximum-risk correlation does not imply a low false-safe rate",
                 loc="left", fontsize=6.25, fontweight="bold", pad=2.5)
    clean_axes(ax, grid=False)
    panel_label(ax, "d", x=-0.055, y=1.03)

    pde_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PDE_COLORS[pde],
               markeredgecolor="none", markersize=4.3, label=components.PDE_LABELS[pde].replace("--", "-"))
        for pde in components.PDES
    ]
    model_handles = [
        Line2D([0], [0], marker=components.MODEL_MARKERS[model], linestyle="none",
               markerfacecolor="white", markeredgecolor=COLORS["charcoal"],
               markeredgewidth=0.55, markersize=4.2, label=components.MODEL_LABELS[model])
        for model in components.MODELS
    ]
    pde_legend = ax.legend(
        handles=pde_handles, loc="upper left", bbox_to_anchor=(0.425, 1.00),
        ncols=3, fontsize=5.4, columnspacing=0.8, handletextpad=0.25,
        alignment="left",
    )
    ax.add_artist(pde_legend)
    ax.legend(
        handles=model_handles, loc="upper left", bbox_to_anchor=(0.425, 0.77),
        ncols=4, fontsize=5.2, columnspacing=0.65, handletextpad=0.25,
        alignment="left",
    )


def main() -> None:
    records = components.load_records()
    by_pair = {(record["pde"], record["model"]): record for record in records}
    gray_pinn = by_pair[("grayscott", "pinn")]

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, min(6.56, MAX_HEIGHT_IN)))
    outer = fig.add_gridspec(
        4, 1, height_ratios=[1.35, 1.35, 0.82, 0.74], hspace=0.55,
    )

    row_a = outer[0].subgridspec(1, 3, wspace=0.16)
    correlations_a = []
    for column, (functional, title) in enumerate(
        (("mean", "GS-PINN: Mean risk"),
         ("q90", "GS-PINN: 90th percentile risk"),
         ("max", "GS-PINN: Max risk (tail)"))
    ):
        correlations_a.append(
            original_risk_scatter(
                fig.add_subplot(row_a[column]), gray_pinn, functional, title,
                letter="a" if column == 0 else None, show_legend=column == 0,
            )
        )

    row_b = outer[1].subgridspec(1, 3, wspace=0.16)
    representatives = (
        (gray_pinn, "Gray-Scott PINN"),
        (by_pair[("kolmogorov", "deeponet")], "Kolmogorov DeepONet"),
        (by_pair[("burgers", "pinn")], "Burgers PINN"),
    )
    correlations_b = []
    for column, (record, title) in enumerate(representatives):
        fs_rate = 100.0 * float(np.mean(record["false_safe"]))
        correlations_b.append(
            original_risk_scatter(
                fig.add_subplot(row_b[column]), record, "max", f"{title} (max)\nFS={fs_rate:.1f}%",
                letter="b" if column == 0 else None, show_legend=column == 0,
                maximum_labels=True,
            )
        )

    row_c = outer[2].subgridspec(1, 2, wspace=0.31)
    ax_c_left = fig.add_subplot(row_c[0])
    components.quantile_panel(ax_c_left, gray_pinn)
    for artist in list(ax_c_left.texts):
        if artist.get_text() == "c":
            artist.remove()
    panel_label(ax_c_left, "c", x=-0.13, y=1.03)
    components.peak_alignment_panel(fig.add_subplot(row_c[1]), gray_pinn)
    all_pair_panel(fig.add_subplot(outer[3]), records)

    expected_a = (0.9614007296112089, 0.8060479388567692, 0.7986904354415023)
    expected_b = (0.7986904354415023, 0.7105617823863729, 0.9878182916691289)
    assert np.allclose(correlations_a, expected_a, atol=1e-12, rtol=0)
    assert np.allclose(correlations_b, expected_b, atol=1e-12, rtol=0)
    q50_sur = np.percentile(gray_pinn["z_sur_ts"], 50, axis=1)
    q50_hf = np.percentile(gray_pinn["z_hf_ts"], 50, axis=1)
    q50_r = float(np.corrcoef(q50_sur, q50_hf)[0, 1])
    assert np.isclose(q50_r, 0.9426070306809001, atol=1e-12)
    pair_rows = pair_summary_with_intervals(records)
    assert len(pair_rows) == 12
    assert sum(row["false_safe_count"] for row in pair_rows) == 116

    fig.subplots_adjust(left=0.076, right=0.992, top=0.945, bottom=0.055)
    save_composite(fig, OUT)
    print(OUT)
    print("panel a correlations:", correlations_a)
    print("panel b correlations:", correlations_b)
    print("all-pair false-safe count:", sum(row["false_safe_count"] for row in pair_rows))


if __name__ == "__main__":
    main()

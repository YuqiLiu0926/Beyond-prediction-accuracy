"""Tail-resolved agreement and temporal peak-displacement panels for Figure 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    PDE_COLORS,
    clean_axes,
    panel_label,
    save_figure,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "external_data" / "decision_archives" / "discovery"
OUT = ROOT / "outputs" / "figures" / "Figure_3_components.pdf"

PDES = ("burgers", "grayscott", "kolmogorov")
MODELS = ("deeponet", "fno", "pino", "pinn")
PDE_LABELS = {"burgers": "Burgers", "grayscott": "Gray--Scott", "kolmogorov": "Kolmogorov"}
MODEL_LABELS = {"deeponet": "DeepONet", "fno": "FNO", "pino": "PINO", "pinn": "PINN"}
MODEL_MARKERS = {"deeponet": "o", "fno": "s", "pino": "^", "pinn": "D"}
QUANTILES = np.asarray([50, 60, 70, 80, 90, 95, 97.5, 99, 100], float)
N_BOOTSTRAP = 2000


def load_records() -> list[dict]:
    records = []
    for pde in PDES:
        for model in MODELS:
            path = DATA_ROOT / pde / model / "challenge_conditions" / "decision_data.npz"
            data = np.load(path, allow_pickle=True)
            margin_sur = np.asarray(data["margin_sur"], float)
            margin_hf = np.asarray(data["margin_hf"], float)
            z_sur_ts = np.asarray(data["z_sur_ts_all"], float)
            z_hf_ts = np.asarray(data["z_hf_ts_all"], float)
            assert len(margin_sur) == 200
            assert z_sur_ts.shape == z_hf_ts.shape
            assert np.max(np.abs(np.asarray(data["z_sur"], float) - np.max(z_sur_ts, axis=1))) < 1e-10
            assert np.max(np.abs(np.asarray(data["z_hf"], float) - np.max(z_hf_ts, axis=1))) < 1e-10
            records.append(
                {
                    "pde": pde,
                    "model": model,
                    "z_sur_ts": z_sur_ts,
                    "z_hf_ts": z_hf_ts,
                    "false_safe": (margin_sur >= 0) & (margin_hf < 0),
                    "safe_agreement": (margin_sur >= 0) & (margin_hf >= 0),
                }
            )
    assert len(records) == 12
    assert sum(int(np.sum(record["false_safe"])) for record in records) == 116
    return records


def square_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    low = min(float(np.min(x)), float(np.min(y)))
    high = max(float(np.max(x)), float(np.max(y)))
    span = high - low
    return low - 0.06 * span, high + 0.06 * span


def risk_scatter(
    ax,
    z_sur: np.ndarray,
    z_hf: np.ndarray,
    false_safe: np.ndarray,
    label: str,
    letter: str,
) -> float:
    correlation = float(np.corrcoef(z_sur, z_hf)[0, 1])
    other = ~false_safe
    ax.scatter(
        z_sur[other], z_hf[other], s=10, color=COLORS["candidate"],
        alpha=0.34, edgecolors="none",
    )
    ax.scatter(
        z_sur[false_safe], z_hf[false_safe], s=20, color=COLORS["unsafe"],
        marker="D", edgecolor="white", linewidth=0.35, alpha=0.92,
        label=f"False safe (n={int(np.sum(false_safe))})",
    )
    low, high = square_limits(z_sur, z_hf)
    ax.plot([low, high], [low, high], linestyle="--", color=COLORS["mid_gray"], linewidth=0.70)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Surrogate risk $Z_{\mathrm{sur}}$")
    ax.set_ylabel(r"HF risk $Z_{\mathrm{HF}}$")
    ax.text(0.04, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=6.7, fontweight="bold")
    ax.text(
        0.96, 0.94, rf"$r={correlation:.3f}$", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
    )
    clean_axes(ax, grid=False)
    panel_label(ax, letter, x=-0.16, y=1.03)
    return correlation


def _rowwise_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    return numerator / denominator


def quantile_correlation_summary(record: dict) -> list[dict]:
    z_sur = np.column_stack([np.percentile(record["z_sur_ts"], q, axis=1) for q in QUANTILES])
    z_hf = np.column_stack([np.percentile(record["z_hf_ts"], q, axis=1) for q in QUANTILES])
    correlations = np.asarray([np.corrcoef(z_sur[:, i], z_hf[:, i])[0, 1] for i in range(len(QUANTILES))])
    rng = np.random.default_rng(20260713)
    indices = rng.integers(0, len(z_sur), size=(N_BOOTSTRAP, len(z_sur)))
    boot = np.empty((N_BOOTSTRAP, len(QUANTILES)), float)
    for i in range(len(QUANTILES)):
        boot[:, i] = _rowwise_correlation(z_sur[indices, i], z_hf[indices, i])
    lower, upper = np.percentile(boot, [2.5, 97.5], axis=0)
    return [
        {
            "quantile_percent": float(q),
            "pearson_r": float(r),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "n_controls": len(z_sur),
            "n_bootstrap": N_BOOTSTRAP,
        }
        for q, r, lo, hi in zip(QUANTILES, correlations, lower, upper)
    ]


def peak_displacement(record: dict) -> np.ndarray:
    n_intervals = record["z_sur_ts"].shape[1] - 1
    peak_sur = np.argmax(record["z_sur_ts"], axis=1)
    peak_hf = np.argmax(record["z_hf_ts"], axis=1)
    return 100.0 * np.abs(peak_hf - peak_sur) / n_intervals


def peak_survival_summary(record: dict) -> list[dict]:
    displacement = peak_displacement(record)
    tolerance = np.arange(0, 101, dtype=float)
    rng = np.random.default_rng(20260714)
    rows = []
    for label, mask in (("False safe", record["false_safe"]), ("Safe agreement", record["safe_agreement"])):
        values = displacement[mask]
        survival = 100.0 * np.mean(values[:, None] > tolerance[None, :], axis=0)
        indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
        boot_values = values[indices]
        boot = 100.0 * np.mean(boot_values[:, :, None] > tolerance[None, None, :], axis=1)
        lower, upper = np.percentile(boot, [2.5, 97.5], axis=0)
        for d, value, lo, hi in zip(tolerance, survival, lower, upper):
            rows.append(
                {
                    "group": label,
                    "tolerance_percent_horizon": float(d),
                    "fraction_exceeding_percent": float(value),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "n_controls": len(values),
                    "n_bootstrap": N_BOOTSTRAP,
                }
            )
    return rows


def quantile_panel(ax, record: dict) -> None:
    rows = quantile_correlation_summary(record)
    q = np.asarray([row["quantile_percent"] for row in rows])
    r = np.asarray([row["pearson_r"] for row in rows])
    lo = np.asarray([row["ci95_low"] for row in rows])
    hi = np.asarray([row["ci95_high"] for row in rows])
    mean_r = float(np.corrcoef(np.mean(record["z_sur_ts"], axis=1), np.mean(record["z_hf_ts"], axis=1))[0, 1])

    ax.fill_between(q, lo, hi, color=COLORS["light_blue"], linewidth=0)
    ax.plot(q, r, color=COLORS["candidate"], linewidth=1.35, marker="o", markersize=2.8)
    ax.axhline(mean_r, color=COLORS["mid_gray"], linestyle=(0, (3, 2)), linewidth=0.65)
    ax.text(
        51,
        mean_r + 0.008,
        rf"temporal mean reference $r={mean_r:.3f}$",
        fontsize=5.2,
        color=COLORS["unavailable"],
    )
    ax.scatter([100], [r[-1]], marker="D", s=24, color=COLORS["certificate"], edgecolor="white", linewidth=0.35, zorder=4)
    ax.text(99.0, r[-1] - 0.025, rf"max $r={r[-1]:.3f}$", ha="right", va="top", fontsize=5.5)
    ax.text(90, r[4] + 0.018, rf"$Q_{{90}}$ {r[4]:.3f}", ha="center", fontsize=5.4, color=COLORS["candidate"])
    ax.set_xlim(49, 101)
    ax.set_ylim(0.68, 1.005)
    ax.set_xticks([50, 70, 90, 100], ["50", "70", "90", "Max"])
    ax.set_xlabel("Temporal risk quantile (%)")
    ax.set_ylabel("Surrogate--HF Pearson $r$")
    ax.set_title("Quantile-resolved agreement", loc="left", fontsize=6.2, fontweight="bold", pad=3)
    clean_axes(ax, grid=False)
    panel_label(ax, "c", x=-0.25, y=1.03)


def peak_alignment_panel(ax, record: dict) -> None:
    rows = peak_survival_summary(record)
    for label, color in (("Safe agreement", COLORS["candidate"]), ("False safe", COLORS["unsafe"])):
        group = [row for row in rows if row["group"] == label]
        x = np.asarray([row["tolerance_percent_horizon"] for row in group])
        y = np.asarray([row["fraction_exceeding_percent"] for row in group])
        lo = np.asarray([row["ci95_low"] for row in group])
        hi = np.asarray([row["ci95_high"] for row in group])
        ax.fill_between(x, lo, hi, step="post", color=color, alpha=0.11, linewidth=0)
        ax.step(x, y, where="post", color=color, linewidth=1.15, label=f"{label} (n={group[0]['n_controls']})")

    displacement = peak_displacement(record)
    fs = displacement[record["false_safe"]]
    sa = displacement[record["safe_agreement"]]
    ax.text(
        0.98, 0.93,
        f"median [IQR]\nFS  {np.median(fs):.0f} [{np.percentile(fs,25):.0f}--{np.percentile(fs,75):.0f}]%\n"
        f"SA  {np.median(sa):.0f} [{np.percentile(sa,25):.0f}--{np.percentile(sa,75):.0f}]%",
        transform=ax.transAxes, ha="right", va="top", fontsize=5.3,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 102)
    ax.set_xlabel(r"Tolerance $d$ (% horizon)")
    ax.set_ylabel("Peak-displacement exceedance (%)")
    ax.set_title("Temporal peak mismatch", loc="left", fontsize=6.2, fontweight="bold", pad=3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.63), fontsize=5.2, handlelength=1.4)
    clean_axes(ax, grid=False)


def all_pair_summary(ax, records: list[dict]) -> None:
    correlations = []
    false_safe_rates = []
    for record in records:
        z_sur = np.max(record["z_sur_ts"], axis=1)
        z_hf = np.max(record["z_hf_ts"], axis=1)
        correlation = float(np.corrcoef(z_sur, z_hf)[0, 1])
        fs_rate = 100.0 * float(np.mean(record["false_safe"]))
        correlations.append(correlation)
        false_safe_rates.append(fs_rate)
        ax.scatter(
            correlation, fs_rate, s=37, marker=MODEL_MARKERS[record["model"]],
            facecolor=PDE_COLORS[record["pde"]], edgecolor="white", linewidth=0.45, zorder=3,
        )

    rho, p_value = spearmanr(correlations, false_safe_rates)
    ax.axhline(0, color=COLORS["charcoal"], linewidth=0.65)
    ax.set_xlim(min(correlations) - 0.05, 1.035)
    ax.set_ylim(-1.2, 24.8)
    ax.set_xlabel(r"Within-pair maximum-risk correlation $r_{\max}$")
    ax.set_ylabel("False-safe rate (%)")
    ax.text(
        0.015, 0.94, rf"Across 12 pairs: Spearman $\rho={rho:.2f}$, $P={p_value:.2f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.2,
    )
    clean_axes(ax, grid=False)
    panel_label(ax, "d", x=-0.055, y=1.03)

    pde_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PDE_COLORS[pde],
               markeredgecolor="none", markersize=4.5, label=PDE_LABELS[pde])
        for pde in PDES
    ]
    model_handles = [
        Line2D([0], [0], marker=MODEL_MARKERS[model], linestyle="none", markerfacecolor="white",
               markeredgecolor=COLORS["charcoal"], markersize=4.5, label=MODEL_LABELS[model])
        for model in MODELS
    ]
    first = ax.legend(handles=pde_handles, loc="upper center", bbox_to_anchor=(0.55, 1.01),
                      ncols=3, fontsize=5.9, columnspacing=0.9, handletextpad=0.3)
    ax.add_artist(first)
    ax.legend(handles=model_handles, loc="lower center", bbox_to_anchor=(0.55, -0.02),
              ncols=4, fontsize=5.8, columnspacing=0.9, handletextpad=0.3)


def main() -> None:
    records = load_records()
    gray_pinn = next(record for record in records if record["pde"] == "grayscott" and record["model"] == "pinn")

    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 4.10))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.66], hspace=0.40)
    top = outer[0].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.05, 1.05], wspace=0.52)

    mean_sur = np.mean(gray_pinn["z_sur_ts"], axis=1)
    mean_hf = np.mean(gray_pinn["z_hf_ts"], axis=1)
    q90_sur = np.percentile(gray_pinn["z_sur_ts"], 90, axis=1)
    q90_hf = np.percentile(gray_pinn["z_hf_ts"], 90, axis=1)
    mean_r = risk_scatter(fig.add_subplot(top[0]), mean_sur, mean_hf, gray_pinn["false_safe"], "Mean risk", "a")
    q90_r = risk_scatter(fig.add_subplot(top[1]), q90_sur, q90_hf, gray_pinn["false_safe"], "90th percentile", "b")
    fig.axes[0].legend(loc="lower right", fontsize=5.4, handletextpad=0.3)
    quantile_panel(fig.add_subplot(top[2]), gray_pinn)
    peak_alignment_panel(fig.add_subplot(top[3]), gray_pinn)
    all_pair_summary(fig.add_subplot(outer[1]), records)

    quantile_rows = quantile_correlation_summary(gray_pinn)
    assert np.isclose(mean_r, 0.9614007296112089, atol=1e-12)
    assert np.isclose(quantile_rows[0]["pearson_r"], 0.9426070306809001, atol=1e-12)
    assert np.isclose(q90_r, 0.8060479388567692, atol=1e-12)
    assert np.isclose(quantile_rows[-1]["pearson_r"], 0.7986904354415023, atol=1e-12)
    displacement = peak_displacement(gray_pinn)
    assert np.median(displacement[gray_pinn["false_safe"]]) == 19.0
    assert np.median(displacement[gray_pinn["safe_agreement"]]) == 0.0

    fig.subplots_adjust(left=0.073, right=0.992, top=0.95, bottom=0.095)
    save_figure(fig, OUT)
    print(OUT)
    print(f"mean r={mean_r:.3f}, q90 r={q90_r:.3f}, max r={quantile_rows[-1]['pearson_r']:.3f}")
    print("peak displacement medians: FS=19% horizon, safe agreement=0%")


if __name__ == "__main__":
    main()

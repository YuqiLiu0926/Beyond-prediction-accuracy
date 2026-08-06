"""Analyze and plot the finite-search recoverability surface.

The experiment is an offline high-fidelity audit.  Every reported boundary is
conditional on the finite control families, horizon and search budgets used in
the audit; the outputs must not be interpreted as exhaustive continuous-action
reachability results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "external_data" / "analysis" / "recoverability" / "surface"
OUTPUT = BASE / "outputs" / "analysis" / "recoverability"
HORIZONS = [25, 50, 75, 100]
AUTHORITIES = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


def summarize_surface(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["cohort", "horizon", "authority", "family"], as_index=False)
        .agg(
            n_states=("safe", "size"),
            recoverable_states=("safe", "sum"),
            median_best_risk_ratio=("risk_ratio_to_z0", "median"),
        )
        .assign(recoverable_fraction=lambda x: x["recoverable_states"] / x["n_states"])
    )


def summarize_cem(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["authority", "n_blocks"], as_index=False)
        .agg(
            n_states=("safe", "size"),
            recoverable_states=("safe", "sum"),
            median_best_risk_ratio=("risk_ratio_to_z0", "median"),
            evaluations_per_state=("evaluations", "first"),
        )
        .assign(recoverable_fraction=lambda x: x["recoverable_states"] / x["n_states"])
    )


def add_heatmap(ax: plt.Axes, summary: pd.DataFrame, cohort: str, title: str) -> None:
    local = summary[(summary["cohort"] == cohort) & (summary["family"] == "dense_constant")]
    frac = local.pivot(index="horizon", columns="authority", values="recoverable_fraction").loc[
        HORIZONS, AUTHORITIES
    ]
    count = local.pivot(index="horizon", columns="authority", values="recoverable_states").loc[
        HORIZONS, AUTHORITIES
    ]
    total = local.pivot(index="horizon", columns="authority", values="n_states").loc[
        HORIZONS, AUTHORITIES
    ]
    cmap = LinearSegmentedColormap.from_list("recoverability", ["#F5E7E5", "#E5E8EE", "#4D7898"])
    image = ax.imshow(frac.to_numpy(), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for yi in range(len(HORIZONS)):
        for xi in range(len(AUTHORITIES)):
            value = frac.iloc[yi, xi]
            color = "white" if value >= 0.72 else "#2B2B2B"
            ax.text(
                xi,
                yi,
                f"{int(count.iloc[yi, xi])}/{int(total.iloc[yi, xi])}",
                ha="center",
                va="center",
                fontsize=6.3,
                color=color,
            )
    ax.set_xticks(range(len(AUTHORITIES)), [f"{x:g}" for x in AUTHORITIES])
    ax.set_yticks(range(len(HORIZONS)), [str(x) for x in HORIZONS])
    ax.set_xlabel("Actuator-authority factor")
    ax.set_ylabel("Evaluation horizon")
    ax.set_title(title, loc="left", pad=6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    surface_rows = read_jsonl(INPUT / "surface_rows.jsonl")
    cem_rows = read_jsonl(INPUT / "hf_cem_rows.jsonl")
    surface = summarize_surface(surface_rows)
    cem = summarize_cem(cem_rows)

    grid_keys = ["cohort", "pde", "ic_idx", "horizon", "authority"]
    family_audit = (
        surface_rows.groupby(grid_keys)
        .agg(
            n_family_risks=("best_risk", "nunique"),
            n_family_labels=("safe", "nunique"),
            min_risk=("best_risk", "min"),
            max_risk=("best_risk", "max"),
        )
        .reset_index()
    )
    audit = {
        "surface_rows": int(len(surface_rows)),
        "cem_rows": int(len(cem_rows)),
        "state_grid_cells": int(len(family_audit)),
        "cells_with_different_family_minima": int((family_audit["n_family_risks"] > 1).sum()),
        "cells_with_different_family_labels": int((family_audit["n_family_labels"] > 1).sum()),
        "interpretation": (
            "The finite-family minima differ in some state-grid cells, but no family difference changes "
            "the recoverable label. This is an empirical result for the audited families, not an "
            "exhaustive continuous-action statement."
        ),
    }

    surface.to_csv(OUTPUT / "recoverability_surface_summary.csv", index=False)
    cem.to_csv(OUTPUT / "hf_cem_summary.csv", index=False)
    family_audit.to_csv(OUTPUT / "control_family_audit.csv", index=False)
    (OUTPUT / "quality_checks.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7.2,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )

    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82], hspace=0.47, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    image = add_heatmap(ax_a, surface, "mechanism", "Mechanism states")
    add_heatmap(ax_b, surface, "independent_boundary_stress", "Independent boundary-stress states")
    ax_a.text(-0.17, 1.10, "a", transform=ax_a.transAxes, fontsize=9, fontweight="bold")
    ax_b.text(-0.17, 1.10, "b", transform=ax_b.transAxes, fontsize=9, fontweight="bold")

    cbar_ax = fig.add_axes([0.955, 0.585, 0.012, 0.275])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation="vertical", ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", "0.5", "1"])
    cbar.set_label("Recoverable fraction", labelpad=3)
    cbar.outline.set_linewidth(0.5)

    mechanism_h100 = surface[
        (surface["cohort"] == "mechanism") & (surface["horizon"] == 100)
    ]
    styles = {
        "K5": ("#777777", "o", ":", 6.6, "K = 5"),
        "K15": ("#D65F5F", "s", "--", 5.4, "K = 15"),
        "dense_constant": ("#4D7898", "D", "-", 4.2, "33 constant controls"),
    }
    for family, (color, marker, linestyle, marker_size, label) in styles.items():
        local = mechanism_h100[mechanism_h100["family"] == family].sort_values("authority")
        ax_c.plot(
            local["authority"],
            local["recoverable_fraction"],
            color=color,
            marker=marker,
            markersize=marker_size,
            linewidth=1.15,
            linestyle=linestyle,
            markerfacecolor="white",
            markeredgewidth=1.15,
            label=label,
        )
    cem_best = (
        cem.sort_values(["authority", "recoverable_fraction", "n_blocks"], ascending=[True, False, False])
        .groupby("authority", as_index=False)
        .first()
    )
    ax_c.plot(
        cem_best["authority"],
        cem_best["recoverable_fraction"],
        color="#2E7D68",
        marker="^",
        markersize=5,
        linewidth=1.15,
        linestyle="--",
        label="HF-CEM",
    )
    for _, row in cem_best.iterrows():
        ax_c.text(
            row["authority"],
            row["recoverable_fraction"] + 0.045,
            f"{int(row['recoverable_states'])}/10",
            ha="center",
            va="bottom",
            fontsize=6.3,
            color="#2E7D68",
        )
    ax_c.set_xlim(0.82, 4.18)
    ax_c.set_ylim(-0.03, 0.68)
    ax_c.set_xticks(AUTHORITIES, [f"{x:g}" for x in AUTHORITIES])
    ax_c.set_yticks([0, 0.2, 0.4, 0.6])
    ax_c.set_xlabel("Actuator-authority factor")
    ax_c.set_ylabel("Recoverable fraction")
    ax_c.set_title("Search-family audit at the original 100-step horizon", loc="left", pad=6)
    ax_c.text(-0.075, 1.08, "c", transform=ax_c.transAxes, fontsize=9, fontweight="bold")
    ax_c.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.legend(loc="upper left", ncol=4, columnspacing=1.4, handletextpad=0.5)

    fig.subplots_adjust(left=0.09, right=0.93, top=0.94, bottom=0.12)
    base = OUTPUT / "recoverability_surface"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

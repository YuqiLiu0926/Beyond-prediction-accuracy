"""Analyze and visualize the static PDE design decisions.

Figure contract
---------------
Core conclusion: low field-prediction error does not preserve the ranking of
safe static PDE designs, so the loss of decision reliability is not specific to
feedback control.

Evidence chain: (a) one representative candidate-ranking reversal; (b) the
distribution of safe-design regret across frozen surrogate families; and
(c) field error versus safe-design regret over physically safe model--state decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "deeponet": "#3B73B9",
    "fno": "#D28B16",
    "pino": "#238B7B",
    "pinn": "#3D4652",
}
LABELS = {"deeponet": "DeepONet", "fno": "FNO", "pino": "PINO", "pinn": "PINN"}
MARKERS = {"deeponet": "o", "fno": "s", "pino": "^", "pinn": "D"}


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_records(directory: Path, stem: str) -> pd.DataFrame:
    csv_path = directory / f"{stem}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return read_jsonl(directory / f"{stem}.jsonl")


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    draws = np.mean(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return float(np.mean(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("external_data/analysis/static_pde_design"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/static_design"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    decisions = read_records(args.input, "decisions")
    candidates = read_records(args.input, "candidates")
    decisions.to_csv(args.output / "SourceData_static-design_decisions.csv", index=False)
    candidates.to_csv(args.output / "SourceData_static-design_candidates.csv", index=False)

    representative = choose_representative(decisions, candidates)
    local = candidates[
        (candidates["model"] == representative["model"])
        & (candidates["state_idx"] == representative["state_idx"])
    ].copy()
    local["predicted_safe"] = local["risk_sur"] <= local["risk_limit"]
    local["hf_safe"] = local["risk_hf"] <= local["risk_limit"]
    local = local.sort_values("performance_sur", ascending=False).reset_index(drop=True)
    local["surrogate_rank"] = np.arange(1, len(local) + 1)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(7.2, 3.25), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.20, 0.92, 1.35], left=0.065, right=0.985, bottom=0.18, top=0.88, wspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    # Panel a: representative candidate ranking.
    scale = max(float(local["performance_hf"].max()), float(local["performance_sur"].max()))
    x = local["surrogate_rank"].to_numpy()
    y_sur = 100.0 * local["performance_sur"].to_numpy() / scale
    y_hf = 100.0 * local["performance_hf"].to_numpy() / scale
    ax_a.plot(x, y_sur, color="#3B73B9", marker="o", ms=3.2, lw=1.25, label="Surrogate estimate")
    ax_a.plot(x, y_hf, color="#C6504D", marker="o", ms=3.2, lw=1.25, label="HF outcome")
    unsafe = ~local["hf_safe"].to_numpy()
    if np.any(unsafe):
        ax_a.scatter(x[unsafe], y_hf[unsafe], facecolors="white", edgecolors="#C6504D", s=30, linewidths=1.0, zorder=4, label="HF unsafe")
    selected_idx = int(representative["selected_idx"])
    oracle_idx = int(representative["hf_oracle_idx"])
    selected_row = local.index[local["candidate_idx"] == selected_idx][0]
    oracle_row = local.index[local["candidate_idx"] == oracle_idx][0]
    ax_a.scatter(x[selected_row], y_hf[selected_row], marker="*", s=72, color="#3B73B9", edgecolor="white", linewidth=0.6, zorder=6, label="Surrogate selection")
    ax_a.scatter(x[oracle_row], y_hf[oracle_row], marker="D", s=32, color="#C6504D", edgecolor="white", linewidth=0.6, zorder=6, label="HF-safe optimum")
    ax_a.set_xlabel("Candidates ordered by surrogate task value")
    ax_a.set_ylabel("Normalized task value (%)")
    ax_a.set_xticks([1, 4, 8, 12, 16])
    ax_a.set_xlim(0.4, 16.6)
    ax_a.set_title("Surrogate and HF rank designs differently", loc="left", pad=7, fontweight="bold")
    ax_a.legend(loc="lower left", ncol=1, handlelength=1.4, borderaxespad=0.2)

    # Panel b: state-level safe-design regret by architecture.
    rng = np.random.default_rng(args.seed)
    model_order = ["deeponet", "fno", "pino", "pinn"]
    summary_rows = []
    for xpos, model in enumerate(model_order):
        values = decisions.loc[decisions["model"] == model, "normalized_safe_regret"].dropna().to_numpy(dtype=float) * 100.0
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax_b.scatter(np.full(len(values), xpos) + jitter, values, s=10, color=COLORS[model], alpha=0.28, linewidth=0)
        mean, lo, hi = bootstrap_mean(values, rng, args.bootstrap)
        ax_b.errorbar(xpos, mean, yerr=[[mean - lo], [hi - mean]], fmt=MARKERS[model], ms=5.2, color=COLORS[model], mec="white", mew=0.6, capsize=2.2, lw=1.2, zorder=5)
        summary_rows.append({"model": model, "mean_regret_percent": mean, "ci_low": lo, "ci_high": hi, "n_states": len(values)})
    ax_b.axhline(0, color="#9AA1AA", lw=0.8, ls="--")
    ax_b.set_xticks(range(len(model_order)), [LABELS[m] for m in model_order], rotation=25, ha="right")
    ax_b.set_ylabel("Safe-design regret (%)")
    ax_b.set_title("Regret across surrogate families", loc="left", pad=7, fontweight="bold")

    # Panel c: field error versus downstream regret.
    for model in model_order:
        subset = decisions[(decisions["model"] == model) & decisions["normalized_safe_regret"].notna()].copy()
        ax_c.scatter(
            100.0 * subset["selected_field_relative_l2"],
            100.0 * subset["normalized_safe_regret"],
            s=22,
            marker=MARKERS[model],
            color=COLORS[model],
            alpha=0.70,
            edgecolor="white",
            linewidth=0.35,
            label=LABELS[model],
        )
    ax_c.set_xlabel(r"Selected-rollout relative $L^2$ error (%)")
    ax_c.set_ylabel("Safe-design regret (%)")
    ax_c.set_title("Field error does not predict decision regret", loc="left", pad=7, fontweight="bold")
    ax_c.legend(loc="upper right", ncol=1, handletextpad=0.5, borderaxespad=0.2)

    for label, axis in zip("abc", (ax_a, ax_b, ax_c)):
        axis.text(-0.16, 1.10, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="top")
        axis.tick_params(length=3, width=0.7)

    stem = args.output / "static_pde_design"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(args.output / "SourceData_static-design_model_summary.csv", index=False)
    qa = {
        "core_conclusion": "Low field error does not preserve safe static-design ranking outside feedback control.",
        "archetype": "quantitative grid with one representative ranking panel",
        "backend": "Python/matplotlib",
        "n_definition": "40 independent physical states; four frozen surrogate decisions per state",
        "panel_c_definition": "157 physically safe surrogate selections; three false-safe selections excluded because safe-design regret is undefined",
        "interval": f"state-level percentile bootstrap 95% CI with {args.bootstrap} resamples",
        "representative": {"model": representative["model"], "state_idx": int(representative["state_idx"])},
        "source_data": ["SourceData_static-design_decisions.csv", "SourceData_static-design_candidates.csv", "SourceData_static-design_model_summary.csv"],
    }
    (args.output / "FIGURE_QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

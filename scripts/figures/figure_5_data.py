"""Load Gray--Scott closed-loop evidence for Figure 5.

The figure is built from episode-level HF traces for three frozen seeds.  It
shows the temporal endpoint, paired safety change, task value and execution
rate without reducing the evidence to pooled bar charts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_style import COLORS, DOUBLE_COLUMN_IN, clean_axes, panel_label


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "external_data" / "closed_loop" / "gray_scott"
FIGURE_DIR = ROOT / "outputs" / "figures"
OUT_STEM = FIGURE_DIR / "Figure_5_data_check"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_bundle(fig: plt.Figure) -> None:
    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".svg"))
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=600)
    fig.savefig(OUT_STEM.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def prepare_data() -> dict:
    summary = read_json(DATA_DIR / "independent_evaluation_summary.json")
    protocol = read_json(DATA_DIR / "protocol.json")
    seeds = [int(seed) for seed in summary["seeds"]]
    frozen = [
        int(seed)
        for seed in protocol["state_selection"]["independent_evaluation_indices"]
    ]
    assert summary["status"] == "PASS"
    assert summary["no_pooling_rule_enforced"] is True
    assert seeds == frozen
    assert int(protocol["state_selection"]["development_seed"]) not in seeds

    episode_pairs = []
    for set_index, seed in enumerate(seeds, start=1):
        rows = read_jsonl(DATA_DIR / f"independent_set_{set_index}.jsonl")
        by_key = {(row["mode"], int(row["ic_idx"])): row for row in rows}
        assert len(rows) == 48
        assert len(by_key) == 48
        indices = sorted(int(row["ic_idx"]) for row in rows if row["mode"] == "raw")
        assert len(indices) == 16
        for ic_idx in indices:
            raw = by_key[("raw", ic_idx)]
            certified = by_key[("continuation_preserving", ic_idx)]
            continuation = by_key[("continuation_only", ic_idx)]
            assert len(raw["risk_trace"]) == len(certified["risk_trace"]) == 61
            assert math.isclose(float(raw["boundary"]), float(certified["boundary"]), rel_tol=0, abs_tol=1e-12)
            assert int(certified["unsafe"]) == 0
            episode_pairs.append(
                {
                    "seed": seed,
                    "ic_idx": ic_idx,
                    "raw": raw,
                    "certified": certified,
                    "continuation": continuation,
                }
            )

    seed_summaries = {
        seed: read_json(DATA_DIR / f"independent_set_{set_index}_summary.json")
        for set_index, seed in enumerate(seeds, start=1)
    }
    task_rows = {}
    for seed, seed_summary in seed_summaries.items():
        assert int(seed_summary["ic_seed"]) == seed
        for row in seed_summary["paired_controller_vs_continuation_only"]["rows"]:
            task_rows[(seed, int(row["ic_idx"]))] = row
    assert len(task_rows) == len(episode_pairs) == 48
    for pair in episode_pairs:
        pair["task"] = task_rows[(pair["seed"], pair["ic_idx"])]
        assert math.isclose(
            float(pair["task"]["raw_J"]),
            float(
                next(
                    row["J_hf_episode"]
                    for row in seed_summaries[pair["seed"]]["audited_rows"]
                    if int(row["ic_idx"]) == pair["ic_idx"] and row["mode"] == "raw"
                )
            ),
            rel_tol=0,
            abs_tol=1e-12,
        )

    raw_unsafe = sum(int(pair["raw"]["unsafe"]) for pair in episode_pairs)
    certified_unsafe = sum(int(pair["certified"]["unsafe"]) for pair in episode_pairs)
    releases = sum(int(pair["certified"]["candidate_releases"]) for pair in episode_pairs)
    decisions = sum(len(pair["certified"]["decisions"]) for pair in episode_pairs)
    assert (raw_unsafe, certified_unsafe, releases, decisions) == (36, 0, 249, 288)
    assert math.isclose(summary["by_mode"]["continuation_preserving"]["candidate_release_fraction"],
                        releases / decisions, rel_tol=0, abs_tol=1e-12)

    return {
        "summary": summary,
        "seed_summaries": seed_summaries,
        "seeds": seeds,
        "pairs": episode_pairs,
        "raw_unsafe": raw_unsafe,
        "certified_unsafe": certified_unsafe,
        "releases": releases,
        "decisions": decisions,
    }


def trace_panel(ax, data: dict) -> None:
    raw = np.asarray([
        np.asarray(pair["raw"]["risk_trace"], float) / float(pair["raw"]["boundary"])
        for pair in data["pairs"]
    ])
    certified = np.asarray([
        np.asarray(pair["certified"]["risk_trace"], float) / float(pair["certified"]["boundary"])
        for pair in data["pairs"]
    ])
    steps = np.arange(raw.shape[1])

    # Thin episode traces preserve outcome heterogeneity; quantiles provide the trend.
    for values in raw:
        ax.plot(steps, values, color=COLORS["unsafe"], linewidth=0.35, alpha=0.12, zorder=1)
    for values in certified:
        ax.plot(steps, values, color=COLORS["controller"], linewidth=0.35, alpha=0.12, zorder=1)
    for values, color, label in (
        (raw, COLORS["unsafe"], "Raw surrogate"),
        (certified, COLORS["controller"], "Certified controller"),
    ):
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
        ax.fill_between(steps, q25, q75, color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(steps, median, color=color, linewidth=1.35, label=label, zorder=3)

    ax.axhline(1.0, color=COLORS["charcoal"], linestyle=(0, (3, 2)), linewidth=0.8)
    for step in range(10, 61, 10):
        ax.axvline(step, color=COLORS["light_gray"], linewidth=0.45, zorder=0)
    ax.text(59.5, 1.007, "safety limit", ha="right", va="bottom", fontsize=5.5)
    ax.text(0.02, 0.96, "36/48 crossed", transform=ax.transAxes, color=COLORS["unsafe"],
            fontsize=5.8, va="top")
    ax.text(0.02, 0.87, "0/48 crossed", transform=ax.transAxes, color=COLORS["controller"],
            fontsize=5.8, va="top")
    ax.set_xlabel("Closed-loop step")
    ax.set_ylabel(r"Normalized HF risk, $Z_{\rm HF}/z_{\rm lim}$")
    ax.set_xlim(0, 60)
    ax.set_ylim(min(0.68, float(np.min(certified)) - 0.02), max(1.12, float(np.max(raw)) + 0.015))
    ax.legend(loc="lower left", ncol=2, columnspacing=0.9, handlelength=1.5)
    clean_axes(ax, grid=False)
    panel_label(ax, "a", x=-0.12)


def paired_peak_panel(ax, data: dict) -> None:
    for pair in data["pairs"]:
        raw = float(pair["raw"]["max_ratio_to_boundary"])
        certified = float(pair["certified"]["max_ratio_to_boundary"])
        line_color = COLORS["unsafe"] if int(pair["raw"]["unsafe"]) else COLORS["mid_gray"]
        ax.plot([0, 1], [raw, certified], color=line_color, alpha=0.34, linewidth=0.55, zorder=1)
        ax.scatter(0, raw, s=11, facecolor=line_color, edgecolor="none", alpha=0.70, zorder=2)
        ax.scatter(1, certified, s=12, facecolor=COLORS["controller"], edgecolor="white",
                   linewidth=0.25, alpha=0.82, zorder=3)
    ax.axhline(1.0, color=COLORS["charcoal"], linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(0.98, 1.006, "safety limit", ha="right", va="bottom", fontsize=5.4)
    ax.set_xlim(-0.28, 1.28)
    ax.set_xticks([0, 1], ["Raw", "Certified"])
    ax.set_ylabel("Peak normalized HF risk")
    peak_max = max(float(pair["raw"]["max_ratio_to_boundary"]) for pair in data["pairs"])
    ax.set_ylim(0.72, max(1.12, peak_max + 0.01))
    clean_axes(ax, grid=False)
    panel_label(ax, "b", x=-0.22)


def task_panel(ax, data: dict) -> None:
    seed_to_x = {seed: index for index, seed in enumerate(data["seeds"])}
    rng = np.random.default_rng(20260713)
    for pair in data["pairs"]:
        xpos = seed_to_x[pair["seed"]] + rng.uniform(-0.14, 0.14)
        ratio = float(pair["task"]["controller_to_continuation_only_ratio"])
        ax.scatter(xpos, ratio, s=14, facecolor=COLORS["candidate"], edgecolor="white",
                   linewidth=0.25, alpha=0.72, zorder=2)
    for index, seed in enumerate(data["seeds"]):
        values = np.asarray([
            float(pair["task"]["controller_to_continuation_only_ratio"])
            for pair in data["pairs"] if pair["seed"] == seed
        ])
        ax.plot([index - 0.18, index + 0.18], [np.median(values)] * 2,
                color=COLORS["charcoal"], linewidth=1.2, zorder=3)

    pooled = float(data["summary"]["aggregate_controller_to_continuation_only_mean_J_ratio"])
    ax.scatter(3, pooled, marker="D", s=31, color=COLORS["controller"], zorder=4)
    ax.text(3, pooled * 1.18, f"{pooled:.2f}x", ha="center", fontsize=5.8)
    ax.axhline(1.0, color=COLORS["charcoal"], linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(2.48, 1.10, "Continuation only", ha="right", va="bottom", fontsize=5.4)
    ax.set_yscale("log")
    ax.set_ylim(0.9, 150)
    ax.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax.set_xticks(range(4), ["Seed 1", "Seed 2", "Seed 3", "Pooled"])
    ax.set_ylabel(r"Task value, $J/J_{\rm cont}$")
    clean_axes(ax, grid=False)
    panel_label(ax, "c", x=-0.15)


def execution_panel(ax, data: dict) -> None:
    offsets = np.linspace(-0.14, 0.14, 16)
    seed_means = []
    for index, seed in enumerate(data["seeds"]):
        pairs = [pair for pair in data["pairs"] if pair["seed"] == seed]
        values = np.asarray([
            100.0 * int(pair["certified"]["candidate_releases"]) / len(pair["certified"]["decisions"])
            for pair in pairs
        ])
        seed_means.append(float(np.mean(values)))
        # Sort before assigning offsets so discrete rates remain readable.
        order = np.argsort(values)
        ax.scatter(values[order], index + offsets, s=15, color=COLORS["candidate"],
                   alpha=0.58, edgecolor="white", linewidth=0.25, zorder=2)
        ax.scatter(np.mean(values), index, marker="D", s=34, color=COLORS["controller"],
                   edgecolor="white", linewidth=0.35, zorder=3)

    pooled = 100.0 * data["releases"] / data["decisions"]
    ax.axvline(pooled, color=COLORS["controller"], linestyle=(0, (3, 2)), linewidth=0.9)
    ax.text(pooled - 1.0, 2.48, f"pooled {pooled:.1f}%", ha="right", va="bottom",
            fontsize=5.8, color=COLORS["controller"])
    ax.set_yticks(range(3), ["Seed 1", "Seed 2", "Seed 3"])
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Surrogate block execution rate (%)")
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=COLORS["candidate"], label="Episode"),
        Line2D([], [], marker="D", linestyle="none", color=COLORS["controller"], label="Seed mean"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=2, columnspacing=0.8,
              handletextpad=0.3)
    clean_axes(ax, grid=False)
    panel_label(ax, "d", x=-0.16)


def main() -> None:
    data = prepare_data()
    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 4.85))
    grid = fig.add_gridspec(
        2, 6, height_ratios=[1.08, 0.92], hspace=0.44, wspace=0.68,
        left=0.085, right=0.985, bottom=0.10, top=0.95,
    )
    trace_panel(fig.add_subplot(grid[0, 0:4]), data)
    paired_peak_panel(fig.add_subplot(grid[0, 4:6]), data)
    task_panel(fig.add_subplot(grid[1, 0:3]), data)
    execution_panel(fig.add_subplot(grid[1, 3:6]), data)
    save_bundle(fig)
    print(OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

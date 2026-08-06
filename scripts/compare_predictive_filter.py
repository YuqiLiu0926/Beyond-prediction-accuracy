#!/usr/bin/env python3
"""Audit an assumption-matched full-plan predictive-safety-filter baseline.

The baseline and continuation-preserving controller use the same Gray--Scott
states, surrogate optimizer, safety boundary, numerical levels and one-sided
allowances.  Their only intended difference is the evaluated future-control
object: one optimized full plan versus an optimized block joined to a finite
continuation family.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gray_scott_reference_dynamics import make_evaluator, physics_rollout, risk_state


BASE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    extreme = max(wins, losses)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = sum(int(row["decisions"]) for row in rows)
    return {
        "episodes": len(rows),
        "decisions": decisions,
        "unsafe_episodes": sum(int(row["unsafe"]) for row in rows),
        "unavailable_decisions": sum(int(row["unavailable_decisions"]) for row in rows),
        "candidate_releases": sum(int(row["candidate_releases"]) for row in rows),
        "candidate_release_fraction": sum(int(row["candidate_releases"]) for row in rows)
        / decisions,
        "interventions": sum(int(row["interventions"]) for row in rows),
        "mean_J_hf_episode": float(np.mean([row["J_hf_episode"] for row in rows])),
        "mean_max_ratio_to_boundary": float(
            np.mean([row["max_ratio_to_boundary"] for row in rows])
        ),
        "mean_certificate_seconds_per_decision": float(
            np.mean([row["certificate_seconds_per_decision"] for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-rows",
        type=Path,
        default=BASE
        / "external_data/analysis/predictive_safety_filter/full_plan_episodes.jsonl",
    )
    parser.add_argument(
        "--proposed-summary",
        type=Path,
        default=BASE
        / "external_data/closed_loop/gray_scott/independent_set_1_summary.json",
    )
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=BASE
        / "external_data/analysis/recoverability/sequence_definition_ablation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "outputs/analysis/predictive_filter_comparison",
    )
    parser.add_argument("--ic-seed", type=int, default=20260821)
    parser.add_argument("--n-ics", type=int, default=100)
    parser.add_argument("--model", default="fno")
    parser.add_argument("--planning-horizon", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=BASE / "external_data/decision_archives/evaluation",
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=BASE / "external_data/checkpoints/discovery",
    )
    parser.add_argument(
        "--model-data-root",
        type=Path,
        default=BASE / "external_data/datasets/discovery",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=BASE / "scripts/train_discovery_surrogates.py",
    )
    parser.add_argument(
        "--burgers-script",
        type=Path,
        default=BASE / "scripts/generate_burgers_decisions.py",
    )
    parser.add_argument(
        "--grayscott-script",
        type=Path,
        default=BASE / "scripts/generate_gray_scott_decisions.py",
    )
    parser.add_argument(
        "--kolmogorov-script",
        type=Path,
        default=BASE / "scripts/generate_kolmogorov_decisions.py",
    )
    args = parser.parse_args()

    baseline_raw = [
        json.loads(line)
        for line in args.baseline_rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(baseline_raw) != 16:
        raise RuntimeError(f"Expected 16 baseline episodes, found {len(baseline_raw)}")
    if {row["mode"] for row in baseline_raw} != {"full_plan"}:
        raise RuntimeError("Unexpected baseline mode")

    proposed_payload = json.loads(args.proposed_summary.read_text(encoding="utf-8"))
    proposed_rows = [
        row
        for row in proposed_payload["audited_rows"]
        if row["mode"] == "continuation_preserving"
    ]
    proposed_by_index = {int(row["ic_idx"]): row for row in proposed_rows}

    evaluator = make_evaluator(args)
    ood = np.asarray(evaluator.x0s, dtype=np.float64)
    evaluator.refresh_ics(args.n_ics, "high_gradient")
    base = np.asarray(evaluator.x0s, dtype=np.float64)

    audited: list[dict[str, Any]] = []
    for row in baseline_raw:
        index = int(row["ic_idx"])
        initial = base[index] + float(row["rho"]) * (ood[index] - base[index])
        initial_error = abs(risk_state(evaluator, initial) - float(row["initial_risk"]))
        states, risk = physics_rollout(
            evaluator,
            initial,
            np.asarray(row["controls"], dtype=np.float64),
            "hf",
            1,
        )
        trace_error = float(
            np.max(np.abs(risk - np.asarray(row["risk_trace"], dtype=np.float64)))
        )
        unsafe = int(float(np.max(risk)) > float(row["boundary"]))
        releases = sum(
            decision["source"] == "surrogate_candidate" for decision in row["decisions"]
        )
        if max(initial_error, trace_error) > 1e-8:
            raise RuntimeError((index, initial_error, trace_error))
        if unsafe != int(row["unsafe"]) or releases != int(row["candidate_releases"]):
            raise RuntimeError((index, unsafe, releases))
        n_decisions = len(row["decisions"])
        audited.append(
            {
                "ic_idx": index,
                "mode": "full_plan_filter",
                "unsafe": unsafe,
                "unavailable_decisions": int(row["unavailable_decisions"]),
                "candidate_releases": int(row["candidate_releases"]),
                "interventions": int(row["interventions"]),
                "decisions": n_decisions,
                "J_hf_episode": float(
                    evaluator.mod.nominal_performance(states, evaluator.actuator)
                ),
                "max_ratio_to_boundary": float(np.max(risk) / float(row["boundary"])),
                "certificate_seconds_per_decision": float(
                    row["certificate_seconds"] / n_decisions
                ),
                "hf_replay_trace_error": trace_error,
            }
        )

    baseline_by_index = {int(row["ic_idx"]): row for row in audited}
    if set(baseline_by_index) != set(proposed_by_index):
        raise RuntimeError(
            {
                "missing_baseline": sorted(set(proposed_by_index) - set(baseline_by_index)),
                "missing_proposed": sorted(set(baseline_by_index) - set(proposed_by_index)),
            }
        )

    paired: list[dict[str, Any]] = []
    for index in sorted(baseline_by_index):
        baseline = baseline_by_index[index]
        proposed = proposed_by_index[index]
        paired.append(
            {
                "ic_idx": index,
                "proposed_minus_full_plan_J": float(
                    proposed["J_hf_episode"] - baseline["J_hf_episode"]
                ),
                "proposed_J_hf_episode": float(proposed["J_hf_episode"]),
                "full_plan_J_hf_episode": float(baseline["J_hf_episode"]),
                "proposed_candidate_releases": int(proposed["candidate_releases"]),
                "full_plan_candidate_releases": int(baseline["candidate_releases"]),
                "proposed_unsafe": int(proposed["unsafe"]),
                "full_plan_unsafe": int(baseline["unsafe"]),
            }
        )

    wins = sum(row["proposed_minus_full_plan_J"] > 0 for row in paired)
    losses = sum(row["proposed_minus_full_plan_J"] < 0 for row in paired)
    proposed_summary = summarize(proposed_rows)
    baseline_summary = summarize(audited)
    development = json.loads(args.development_summary.read_text(encoding="utf-8"))
    payload = {
        "experiment": "Predictive safety-filter comparison under matched assumptions",
        "interpretation": (
            "Mechanism-matched baseline, not an exact reproduction of a specific published "
            "predictive safety filter. The paired cohort was independent of controller "
            "development but had been used previously for proposed-controller confirmation."
        ),
        "shared_assumptions": [
            "same measured state and frozen surrogate proposal",
            "same 100-step horizon and 10-step executed block",
            "same state-relative safety limit",
            "same numerical levels and one-sided allowances",
        ],
        "difference": (
            "The full-plan filter evaluates only the optimized plan U*. The proposed "
            "controller evaluates the optimized block joined to each predefined continuation."
        ),
        "paired_evaluation": {
            "ic_seed": args.ic_seed,
            "full_plan_filter": baseline_summary,
            "continuation_preserving": proposed_summary,
            "candidate_release_gain": (
                proposed_summary["candidate_releases"]
                - baseline_summary["candidate_releases"]
            ),
            "mean_performance_difference": (
                proposed_summary["mean_J_hf_episode"]
                - baseline_summary["mean_J_hf_episode"]
            ),
            "paired_performance": {
                "wins": wins,
                "losses": losses,
                "ties": len(paired) - wins - losses,
                "exact_two_sided_sign_p": sign_test_two_sided(wins, losses),
            },
        },
        "development_comparison": {
            "full_plan_filter": development["by_mode"]["full_plan"],
            "continuation_preserving": development["by_mode"]["continuation_preserving"],
        },
        "provenance": {
            "analysis_script_sha256": sha256(Path(__file__)),
            "baseline_controller_sha256": sha256(
                BASE / "scripts" / "compare_sequence_definitions.py"
            ),
            "baseline_rows_sha256": sha256(args.baseline_rows),
            "proposed_summary_sha256": sha256(args.proposed_summary),
            "development_summary_sha256": sha256(args.development_summary),
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output / "paired_episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    with (args.output / "full_plan_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audited[0]))
        writer.writeheader()
        writer.writerows(audited)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

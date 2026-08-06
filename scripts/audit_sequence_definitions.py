#!/usr/bin/env python3
"""Audit complete-sequence definitions in Gray--Scott closed-loop evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gray_scott_reference_dynamics import make_evaluator, physics_rollout, risk_state


BASE = Path(__file__).resolve().parents[1]
ABLATION_MODES = ("full_plan", "block_only")


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


def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    local = [row for row in rows if row["mode"] == mode]
    decisions = sum(row["decisions"] for row in local)
    return {
        "episodes": len(local),
        "unsafe_episodes": sum(row["unsafe"] for row in local),
        "unavailable_decisions": sum(row["unavailable_decisions"] for row in local),
        "candidate_releases": sum(row["candidate_releases"] for row in local),
        "candidate_release_fraction": sum(row["candidate_releases"] for row in local)
        / decisions,
        "interventions": sum(row["interventions"] for row in local),
        "N1_evaluations": sum(row["N1_evaluations"] for row in local),
        "mean_J_hf_episode": float(np.mean([row["J_hf_episode"] for row in local])),
        "mean_max_ratio_to_boundary": float(
            np.mean([row["max_ratio_to_boundary"] for row in local])
        ),
        "mean_certificate_seconds_per_decision": float(
            np.mean([row["certificate_seconds_per_decision"] for row in local])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=BASE / "external_data/analysis/recoverability/sequence_evaluation_ablation",
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=BASE / "external_data/closed_loop/gray_scott",
    )
    parser.add_argument("--pattern", default="part_*.jsonl")
    parser.add_argument("--ic-seed", type=int, default=20260811)
    parser.add_argument(
        "--indices",
        default="82,65,14,96,66,18,52,86,8,73,20,98,42,0,21,56",
    )
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
    protocol_path = args.root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    if indices != [int(value) for value in protocol["development_state_indices"]]:
        raise RuntimeError("Index list differs from frozen protocol")
    paths = sorted(args.root.glob(args.pattern))
    if len(paths) != 4:
        raise RuntimeError(f"Expected four sequence-evaluation partitions, found {paths}")
    rows = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected = {(index, mode) for index in indices for mode in ABLATION_MODES}
    observed = {(int(row["ic_idx"]), str(row["mode"])) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise RuntimeError((len(rows), len(expected), expected - observed, observed - expected))
    ablation_code_hash = sha256(BASE / "scripts/compare_sequence_definitions.py")
    controller_code_hash = sha256(BASE / "scripts/gray_scott_continuation_controller.py")
    if ablation_code_hash != protocol["sequence_evaluation_code_sha256"]:
        raise RuntimeError("Sequence-evaluation code differs from the frozen protocol")
    if controller_code_hash != protocol["continuation_controller_code_sha256"]:
        raise RuntimeError("Continuation-controller code differs from the frozen protocol")

    evaluator = make_evaluator(args)
    ood = np.asarray(evaluator.x0s, dtype=np.float64)
    evaluator.refresh_ics(args.n_ics, "high_gradient")
    base = np.asarray(evaluator.x0s, dtype=np.float64)
    audited: list[dict[str, Any]] = []
    for row in rows:
        index = int(row["ic_idx"])
        initial = base[index] + float(row["rho"]) * (ood[index] - base[index])
        initial_error = abs(risk_state(evaluator, initial) - float(row["initial_risk"]))
        states, risk = physics_rollout(
            evaluator, initial, np.asarray(row["controls"], dtype=np.float64), "hf", 1
        )
        trace_error = float(
            np.max(np.abs(risk - np.asarray(row["risk_trace"], dtype=np.float64)))
        )
        unsafe = int(float(np.max(risk)) > float(row["boundary"]))
        max_ratio = float(np.max(risk) / float(row["boundary"]))
        unsafe_error = unsafe != int(row["unsafe"])
        ratio_error = abs(max_ratio - float(row["max_ratio_to_boundary"]))
        releases = sum(
            decision["source"] == "surrogate_candidate" for decision in row["decisions"]
        )
        if max(initial_error, trace_error) > 1e-8:
            raise RuntimeError((index, row["mode"], initial_error, trace_error))
        if unsafe_error or ratio_error > 1e-8 or releases != int(row["candidate_releases"]):
            raise RuntimeError(
                (index, row["mode"], unsafe_error, ratio_error, releases)
            )
        decisions = len(row["decisions"])
        audited.append(
            {
                "ic_idx": index,
                "mode": row["mode"],
                "unsafe": unsafe,
                "unavailable_decisions": int(row["unavailable_decisions"]),
                "candidate_releases": int(row["candidate_releases"]),
                "interventions": int(row["interventions"]),
                "N1_evaluations": int(row["N1_evaluations"]),
                "decisions": decisions,
                "J_hf_episode": float(
                    evaluator.mod.nominal_performance(states, evaluator.actuator)
                ),
                "max_ratio_to_boundary": max_ratio,
                "certificate_seconds_per_decision": float(
                    row["certificate_seconds"] / decisions
                ),
                "hf_replay_trace_error": trace_error,
            }
        )

    validation = json.loads(
        (args.validation_root / "development_summary.json").read_text(encoding="utf-8")
    )
    reference_rows = validation["audited_rows"]
    combined = reference_rows + audited
    by_mode = {
        mode: summarize(combined, mode)
        for mode in ("raw", "continuation_preserving", *ABLATION_MODES, "continuation_only")
    }
    keyed = {(int(row["ic_idx"]), str(row["mode"])): row for row in combined}
    paired = []
    for index in indices:
        composite = keyed[(index, "continuation_preserving")]
        full = keyed[(index, "full_plan")]
        difference = float(composite["J_hf_episode"] - full["J_hf_episode"])
        paired.append(
            {
                "ic_idx": index,
                "composite_minus_full_J": difference,
                "composite_unsafe": int(composite["unsafe"]),
                "full_candidate_unsafe": int(full["unsafe"]),
            }
        )
    wins = sum(row["composite_minus_full_J"] > 0 for row in paired)
    losses = sum(row["composite_minus_full_J"] < 0 for row in paired)
    release_gain = (
        by_mode["continuation_preserving"]["candidate_releases"]
        - by_mode["full_plan"]["candidate_releases"]
    )
    gates = {
        "full_plan_zero_unsafe": (
            by_mode["full_plan"]["unsafe_episodes"] == 0
        ),
        "full_plan_zero_unavailable": (
            by_mode["full_plan"]["unavailable_decisions"] == 0
        ),
        "continuation_augmented_zero_unsafe": (
            by_mode["continuation_preserving"]["unsafe_episodes"] == 0
        ),
        "composite_recovers_at_least_three_releases": release_gain >= 3,
        "composite_mean_performance_no_worse": (
            by_mode["continuation_preserving"]["mean_J_hf_episode"]
            >= by_mode["full_plan"]["mean_J_hf_episode"]
        ),
        "hf_replay_error_at_most_1e_8": max(
            row["hf_replay_trace_error"] for row in audited
        )
        <= 1e-8,
    }
    payload = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "sequence_evaluation_code_sha256_verified": ablation_code_hash,
        "continuation_controller_code_sha256_verified": controller_code_hash,
        "by_mode": by_mode,
        "composite_minus_full_candidate_releases": release_gain,
        "paired_composite_vs_full_candidate": {
            "wins": wins,
            "losses": losses,
            "ties": len(paired) - wins - losses,
            "exact_two_sided_sign_p": sign_test_two_sided(wins, losses),
            "rows": paired,
        },
        "registered_gates": gates,
        "audited_ablation_rows": audited,
    }
    output = BASE / "outputs/audits/sequence_evaluation_ablation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

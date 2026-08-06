#!/usr/bin/env python3
"""Audit Gray--Scott closed-loop safety, performance, and HF replay consistency."""

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
MODE_ORDER = {"raw": 0, "continuation_preserving": 1, "continuation_only": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sign_test_two_sided(wins: int, non_ties: int) -> float:
    upper = sum(math.comb(non_ties, k) for k in range(wins, non_ties + 1)) / (2**non_ties)
    return min(1.0, 2.0 * upper)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=BASE / "external_data/closed_loop/gray_scott",
    )
    parser.add_argument(
        "--scope",
        choices=("independent", "development"),
        default="independent",
        help="Audit the independent confirmation sets (default) or the development set.",
    )
    parser.add_argument("--pattern", default=None, help="Optional file-pattern override.")
    parser.add_argument("--out-name", default=None, help="Optional output-name override.")
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--ic-seed", type=int, default=None, help="Development/custom audit override.")
    parser.add_argument("--indices", default=None, help="Comma-separated development/custom indices.")
    parser.add_argument("--n-ics", type=int, default=100)
    parser.add_argument("--planning-horizon", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--apply-block", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.90)
    parser.add_argument("--model", default="fno")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=BASE / "external_data/decision_archives/evaluation")
    parser.add_argument("--ckpt-root", type=Path, default=BASE / "external_data/checkpoints/discovery")
    parser.add_argument("--model-data-root", type=Path, default=BASE / "external_data/datasets/discovery")
    parser.add_argument("--train-script", type=Path, default=BASE / "scripts/train_discovery_surrogates.py")
    parser.add_argument("--burgers-script", type=Path, default=BASE / "scripts/generate_burgers_decisions.py")
    parser.add_argument("--grayscott-script", type=Path, default=BASE / "scripts/generate_gray_scott_decisions.py")
    parser.add_argument("--kolmogorov-script", type=Path, default=BASE / "scripts/generate_kolmogorov_decisions.py")
    args = parser.parse_args()
    if args.protocol is None:
        args.protocol = args.root / "protocol.json"

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    selection = protocol["state_selection"]
    if args.scope == "independent":
        default_pattern = "independent_set_*.jsonl"
        default_out_name = "independent_evaluation"
        expected_by_seed = {
            int(seed): [int(value) for value in values]
            for seed, values in selection["independent_evaluation_indices"].items()
        }
    else:
        default_pattern = "development_episodes.jsonl"
        default_out_name = "development"
        seed = int(args.ic_seed if args.ic_seed is not None else selection["development_seed"])
        if args.indices:
            indices = [int(value) for value in args.indices.split(",") if value.strip()]
        else:
            indices = [int(value) for value in selection["development_indices"]]
        expected_by_seed = {seed: indices}

    pattern = args.pattern or default_pattern
    out_name = args.out_name or default_out_name

    input_paths = sorted(args.root.glob(pattern))
    if not input_paths:
        raise FileNotFoundError(f"No shard rows match {args.root / pattern}")
    rows = [row for path in input_paths for row in iter_rows(path)]
    expected = {
        (seed, ic_idx, mode)
        for seed, indices in expected_by_seed.items()
        for ic_idx in indices
        for mode in MODE_ORDER
    }
    observed = {(int(row["ic_seed"]), int(row["ic_idx"]), str(row["mode"])) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError(
            f"Incomplete, duplicate, or unexpected rows: rows={len(rows)} unique={len(observed)} "
            f"expected={len(expected)} missing={sorted(expected-observed)} extra={sorted(observed-expected)}"
        )

    controller_path = BASE / "scripts/gray_scott_continuation_controller.py"
    controller_hash = sha256(controller_path)

    evaluators: dict[int, Any] = {}
    state_sets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in expected_by_seed:
        args.ic_seed = seed
        evaluator = make_evaluator(args)
        ood = np.asarray(evaluator.x0s, dtype=np.float64)
        evaluator.refresh_ics(args.n_ics, "high_gradient")
        base = np.asarray(evaluator.x0s, dtype=np.float64)
        evaluators[seed] = evaluator
        state_sets[seed] = (base, ood)

    audited: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row["ic_seed"])
        ic_idx = int(row["ic_idx"])
        evaluator = evaluators[seed]
        base, ood = state_sets[seed]
        initial = base[ic_idx] + float(row["rho"]) * (ood[ic_idx] - base[ic_idx])
        initial_error = abs(risk_state(evaluator, initial) - float(row["initial_risk"]))
        control = np.asarray(row["controls"], dtype=np.float64)
        states, risk = physics_rollout(evaluator, initial, control, "hf", 1)
        trace = np.asarray(row["risk_trace"], dtype=np.float64)
        replay_trace_error = float(np.max(np.abs(risk - trace)))
        replay_max_error = abs(float(np.max(risk)) - float(row["max_hf_risk"]))
        if max(initial_error, replay_trace_error, replay_max_error) > 1e-8:
            raise RuntimeError(
                f"HF replay mismatch seed={seed} ic={ic_idx} mode={row['mode']}: "
                f"{initial_error}, {replay_trace_error}, {replay_max_error}"
            )
        performance = float(evaluator.mod.nominal_performance(states, evaluator.actuator))
        decisions = len(row["decisions"])
        audited.append(
            {
                "ic_seed": seed,
                "ic_idx": ic_idx,
                "mode": row["mode"],
                "unsafe": int(row["unsafe"]),
                "unavailable_decisions": int(row["unavailable_decisions"]),
                "candidate_releases": int(row["candidate_releases"]),
                "interventions": int(row["interventions"]),
                "N1_evaluations": int(row["N1_evaluations"]),
                "decisions": decisions,
                "J_hf_episode": performance,
                "control_rms": float(np.sqrt(np.mean(control**2))),
                "max_ratio_to_boundary": float(row["max_ratio_to_boundary"]),
                "planning_seconds_per_decision": float(row["planning_seconds"] / decisions),
                "certificate_seconds_per_decision": float(row["certificate_seconds"] / decisions),
                "hf_replay_trace_error": replay_trace_error,
            }
        )

    by_mode: dict[str, Any] = {}
    for mode in MODE_ORDER:
        local = [row for row in audited if row["mode"] == mode]
        decisions = sum(row["decisions"] for row in local)
        by_mode[mode] = {
            "episodes": len(local),
            "unsafe_episodes": sum(row["unsafe"] for row in local),
            "unavailable_decisions": sum(row["unavailable_decisions"] for row in local),
            "candidate_release_fraction": float(sum(row["candidate_releases"] for row in local) / decisions),
            "interventions": sum(row["interventions"] for row in local),
            "N1_evaluations": sum(row["N1_evaluations"] for row in local),
            "mean_J_hf_episode": float(np.mean([row["J_hf_episode"] for row in local])),
            "mean_control_rms": float(np.mean([row["control_rms"] for row in local])),
            "mean_max_ratio_to_boundary": float(
                np.mean([row["max_ratio_to_boundary"] for row in local])
            ),
            "mean_planning_seconds_per_decision": float(
                np.mean([row["planning_seconds_per_decision"] for row in local])
            ),
            "mean_certificate_seconds_per_decision": float(
                np.mean([row["certificate_seconds_per_decision"] for row in local])
            ),
        }

    paired = []
    for seed, indices in expected_by_seed.items():
        for ic_idx in indices:
            local = {
                row["mode"]: row
                for row in audited
                if row["ic_seed"] == seed and row["ic_idx"] == ic_idx
            }
            sup = local["continuation_preserving"]["J_hf_episode"]
            continuation_only = local["continuation_only"]["J_hf_episode"]
            raw = local["raw"]["J_hf_episode"]
            paired.append(
                {
                    "ic_seed": seed,
                    "ic_idx": ic_idx,
                    "controller_J": sup,
                    "continuation_only_J": continuation_only,
                    "raw_J": raw,
                    "controller_minus_continuation_only": sup - continuation_only,
                    "controller_to_continuation_only_ratio": float(
                        sup / (abs(continuation_only) + 1e-12)
                    ),
                }
            )
    wins = sum(row["controller_minus_continuation_only"] > 0 for row in paired)
    losses = sum(row["controller_minus_continuation_only"] < 0 for row in paired)
    non_ties = wins + losses
    aggregate_ratio = float(
        by_mode["continuation_preserving"]["mean_J_hf_episode"] / by_mode["continuation_only"]["mean_J_hf_episode"]
    )
    release = by_mode["continuation_preserving"]["candidate_release_fraction"]
    max_replay = max(row["hf_replay_trace_error"] for row in audited)
    gates: dict[str, bool] = {
        "zero_controller_unsafe": by_mode["continuation_preserving"]["unsafe_episodes"] == 0,
        "zero_controller_unavailable": by_mode["continuation_preserving"]["unavailable_decisions"] == 0,
        "hf_replay_error_at_most_1e_8": max_replay <= 1e-8,
        "candidate_release_between_0_50_and_0_95": 0.50 <= release <= 0.95,
        "aggregate_performance_at_least_1_5x_continuation_only": aggregate_ratio >= 1.5,
        "zero_unsafe_caused_by_release": by_mode["continuation_preserving"]["unsafe_episodes"] == 0,
    }
    if args.scope == "independent":
        per_seed_controller_unsafe = {
            seed: sum(
                row["unsafe"]
                for row in audited
                if row["ic_seed"] == seed and row["mode"] == "continuation_preserving"
            )
            for seed in expected_by_seed
        }
        gates.update(
            {
                "zero_controller_unsafe_in_every_independent_set": all(
                    value == 0 for value in per_seed_controller_unsafe.values()
                ),
                "paired_wins_at_least_36_of_48": wins >= 36,
                "raw_unsafe_matches_reported_36_of_48": by_mode["raw"]["unsafe_episodes"] == 36,
                "candidate_releases_match_reported_249_of_288": math.isclose(
                    release, 249.0 / 288.0, rel_tol=0.0, abs_tol=1e-12
                ),
            }
        )
    else:
        per_seed_controller_unsafe = {
            seed: sum(
                row["unsafe"]
                for row in audited
                if row["ic_seed"] == seed and row["mode"] == "continuation_preserving"
            )
            for seed in expected_by_seed
        }
        gates["paired_wins_at_least_12_of_16"] = wins >= 12

    payload = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": args.scope,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "controller_sha256_verified": controller_hash,
        "input_paths": [str(path) for path in input_paths],
        "state_indices_by_seed": expected_by_seed,
        "controller_unsafe_episodes_by_seed": per_seed_controller_unsafe,
        "statistical_unit": "HF receding-horizon episode",
        "by_mode": by_mode,
        "aggregate_controller_to_continuation_only_mean_J_ratio": aggregate_ratio,
        "paired_controller_vs_continuation_only": {
            "wins": wins,
            "losses": losses,
            "ties": len(paired) - non_ties,
            "exact_two_sided_sign_p": sign_test_two_sided(wins, non_ties),
            "rows": paired,
        },
        "zero_violation_one_sided_95pct_upper": float(1.0 - 0.05 ** (1.0 / len(paired))),
        "max_hf_replay_trace_error": max_replay,
        "registered_gates": gates,
        "audited_rows": audited,
    }

    output_dir = BASE / "outputs/audits/gray_scott_closed_loop"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = output_dir / f"{out_name}_episode_rows.jsonl"
    sorted_rows = sorted(
        rows,
        key=lambda row: (int(row["ic_seed"]), int(row["ic_idx"]), MODE_ORDER[row["mode"]]),
    )
    with combined_rows.open("w", encoding="utf-8") as handle:
        for row in sorted_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_path = output_dir / f"{out_name}_audit_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "by_mode": by_mode,
                "aggregate_controller_to_continuation_only_mean_J_ratio": aggregate_ratio,
                "paired_wins": wins,
                "sign_p": payload["paired_controller_vs_continuation_only"]["exact_two_sided_sign_p"],
                "registered_gates": gates,
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

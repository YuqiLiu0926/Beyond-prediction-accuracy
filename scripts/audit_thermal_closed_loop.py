#!/usr/bin/env python3
"""Audit the final independent thermal closed-loop evaluation records."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "external_data/closed_loop/thermal"
MODES = ("raw", "continuation_preserving", "continuation_only")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exact_two_sided_sign_p(wins: int, losses: int) -> float:
    non_ties = wins + losses
    if non_ties == 0:
        return 1.0
    extreme = max(wins, losses)
    tail = sum(math.comb(non_ties, k) for k in range(extreme, non_ties + 1))
    return min(1.0, 2.0 * tail / (2**non_ties))


def audit_certificate_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    maximum_error = 0.0
    for row in rows:
        for decision in row["decisions"]:
            for field in ("candidate_certificate", "current_continuation"):
                certificate = decision.get(field)
                if not certificate:
                    continue
                count += 1
                level = certificate["level"]
                if level == "N0":
                    expected = float(certificate["z_N0"]) + float(
                        certificate["N0_allowance"]
                    )
                elif level in {"N1", "unavailable"}:
                    expected = (
                        float(certificate["z_N1"])
                        + float(certificate["richardson_gamma"])
                        * float(certificate["resolution_defect"])
                        + float(certificate["N1_allowance"])
                    )
                else:
                    raise ValueError(f"Unknown certificate level: {level}")
                maximum_error = max(
                    maximum_error, abs(expected - float(certificate["z_upper"]))
                )
    return {
        "certificates_checked": count,
        "maximum_upper_risk_identity_error": maximum_error,
    }


def main() -> None:
    protocol_path = ROOT / "protocol.json"
    rows_path = ROOT / "independent_evaluation_episodes.jsonl"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = read_jsonl(rows_path)

    episode_keys = sorted({(int(row["ic_seed"]), int(row["ic_idx"])) for row in rows})
    expected = {(seed, index, mode) for seed, index in episode_keys for mode in MODES}
    observed = {
        (int(row["ic_seed"]), int(row["ic_idx"]), row["mode"]) for row in rows
    }
    if len(rows) != len(expected) or observed != expected:
        raise RuntimeError((len(rows), len(expected), expected - observed, observed - expected))

    expected_seeds = set(protocol["independent_evaluation"]["initial_state_seeds"])
    observed_seeds = {seed for seed, _ in episode_keys}
    episodes_per_set = int(protocol["independent_evaluation"]["episodes_per_set"])
    if observed_seeds != expected_seeds:
        raise RuntimeError((observed_seeds, expected_seeds))
    for seed in expected_seeds:
        if sum(key[0] == seed for key in episode_keys) != episodes_per_set:
            raise RuntimeError(f"Unexpected episode count for evaluation seed {seed}")

    artifacts = {
        "closed_loop_code": BASE / "scripts/run_thermal_closed_loop.py",
        "numerical_certificate_code": BASE / "scripts/thermal_numerical_certificate.py",
        "thermal_dynamics_code": BASE / "scripts/thermal_dynamics.py",
        "selection_code": BASE / "scripts/select_thermal_evaluation_states.py",
        "training_code": BASE / "scripts/train_thermal_surrogate.py",
        "fno_checkpoint": BASE / "external_data/checkpoints/thermal/fno.pt",
        "calibration_record": (
            BASE / "external_data/numerical_qualification/thermal/one_sided_allowance.json"
        ),
        "evaluation_set_1_manifest": (
            BASE / "external_data/metadata/thermal/independent_evaluation_set_1.json"
        ),
        "evaluation_set_2_manifest": (
            BASE / "external_data/metadata/thermal/independent_evaluation_set_2.json"
        ),
    }
    verified_hashes = {key: sha256(path) for key, path in artifacts.items()}

    by_mode: dict[str, Any] = {}
    for mode in MODES:
        local = [row for row in rows if row["mode"] == mode]
        decisions = sum(len(row["decisions"]) for row in local)
        by_mode[mode] = {
            "episodes": len(local),
            "unsafe_episodes": sum(int(row["unsafe"]) for row in local),
            "mean_performance_conversion": float(
                np.mean([float(row["performance_conversion"]) for row in local])
            ),
            "candidate_releases": sum(int(row["candidate_releases"]) for row in local),
            "candidate_release_fraction": float(
                sum(int(row["candidate_releases"]) for row in local) / decisions
            ),
            "interventions": sum(int(row["interventions"]) for row in local),
            "unavailable_decisions": sum(
                int(row["unavailable_decisions"]) for row in local
            ),
            "N1_evaluations": sum(int(row["N1_evaluations"]) for row in local),
            "mean_planning_seconds_per_decision": float(
                sum(float(row["planning_seconds"]) for row in local) / decisions
            ),
            "mean_certificate_seconds_per_decision": float(
                sum(float(row["certificate_seconds"]) for row in local) / decisions
            ),
            "max_hf_replay_state_error": float(
                max(float(row["hf_replay_state_error"]) for row in local)
            ),
        }

    paired = []
    for seed, index in episode_keys:
        local = {
            row["mode"]: row
            for row in rows
            if int(row["ic_seed"]) == seed and int(row["ic_idx"]) == index
        }
        controller = float(local["continuation_preserving"]["performance_conversion"])
        continuation = float(local["continuation_only"]["performance_conversion"])
        paired.append(
            {
                "ic_seed": seed,
                "ic_idx": index,
                "controller_minus_continuation_only": controller - continuation,
                "raw_unsafe": int(local["raw"]["unsafe"]),
                "controller_unsafe": int(local["continuation_preserving"]["unsafe"]),
                "continuation_only_unsafe": int(local["continuation_only"]["unsafe"]),
            }
        )
    wins = sum(row["controller_minus_continuation_only"] > 0 for row in paired)
    losses = sum(row["controller_minus_continuation_only"] < 0 for row in paired)

    controller_mean = by_mode["continuation_preserving"]["mean_performance_conversion"]
    raw_mean = by_mode["raw"]["mean_performance_conversion"]
    continuation_mean = by_mode["continuation_only"]["mean_performance_conversion"]
    performance_retention = controller_mean / raw_mean
    continuation_ratio = controller_mean / continuation_mean
    certificate_audit = audit_certificate_identity(rows)
    gates = {
        "two_independent_sets_of_64": len(episode_keys) == 128,
        "zero_controller_unsafe": by_mode["continuation_preserving"]["unsafe_episodes"] == 0,
        "zero_controller_unavailable": (
            by_mode["continuation_preserving"]["unavailable_decisions"] == 0
        ),
        "hf_replay_error_at_most_1e_8": (
            by_mode["continuation_preserving"]["max_hf_replay_state_error"] <= 1e-8
        ),
        "certificate_identity_error_at_most_1e_12": (
            certificate_audit["maximum_upper_risk_identity_error"] <= 1e-12
        ),
        "candidate_release_count_is_1258": (
            by_mode["continuation_preserving"]["candidate_releases"] == 1258
        ),
        "safety_no_worse_than_continuation_only": (
            by_mode["continuation_preserving"]["unsafe_episodes"]
            <= by_mode["continuation_only"]["unsafe_episodes"]
        ),
    }
    payload = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": str(protocol_path.relative_to(BASE / "external_data")),
        "protocol_sha256": sha256(protocol_path),
        "verified_hashes": verified_hashes,
        "row_count": len(rows),
        "unique_episode_count": len(episode_keys),
        "by_mode": by_mode,
        "performance_retention_relative_to_raw": performance_retention,
        "performance_ratio_to_continuation_only": continuation_ratio,
        "paired_controller_vs_continuation_only": {
            "wins": wins,
            "losses": losses,
            "ties": len(paired) - wins - losses,
            "exact_two_sided_sign_p": exact_two_sided_sign_p(wins, losses),
        },
        "raw_unsafe_episodes_avoided": sum(
            row["raw_unsafe"] and not row["controller_unsafe"] for row in paired
        ),
        "zero_event_one_sided_95_percent_upper": float(
            1.0 - 0.05 ** (1.0 / len(episode_keys))
        ),
        "certificate_identity_audit": certificate_audit,
        "registered_gates": gates,
    }
    output = BASE / "outputs/audits/thermal_closed_loop_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Evaluate the residual detector and finite continuation family."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from mismatch_detection import (
    finite_sample_upper_quantile,
    fit_gap_model,
    residual_feature_matrix,
)


BASE = Path(__file__).resolve().parents[1]
PDES = ("burgers", "grayscott", "kolmogorov")
MODELS = ("deeponet", "fno", "pinn", "pino")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_rows(pattern: str) -> list[dict[str, Any]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if len(paths) != 4:
        raise RuntimeError(f"Expected four replay shards for {pattern}, found {len(paths)}")
    rows = [row for path in paths for row in iter_jsonl(path)]
    rows.sort(key=lambda row: (str(row["pde"]), str(row["model"]), int(row["ic_idx"])))
    return rows


def z0_map(root: Path, n_per_pair: int) -> dict[tuple[str, str, int], float]:
    values = {}
    for pde in PDES:
        for model in MODELS:
            path = root / pde / model / "challenge_conditions" / "decision_data.npz"
            with np.load(path, allow_pickle=True) as data:
                if len(data["z0"]) != n_per_pair:
                    raise RuntimeError(f"Unexpected z0 length in {path}")
                for ic_idx, value in enumerate(data["z0"]):
                    values[(pde, model, ic_idx)] = float(value)
    return values


def rescore_development(
    rows: list[dict[str, Any]], z0_values: dict[tuple[str, str, int], float], scale: float, source: str
) -> list[dict[str, Any]]:
    rescored = []
    for original in rows:
        row = dict(original)
        key = (str(row["pde"]), str(row["model"]), int(row["ic_idx"]))
        z0 = z0_values[key]
        boundary = scale * z0
        z_hf = float(row["canonical_z_hf"])
        row["z0"] = z0
        row["z_limit_phys"] = boundary
        row["m_sur_replay"] = boundary - float(row["z_sur_replay"])
        row["m_hf"] = boundary - z_hf
        row["replay_regime"] = (
            "SA" if row["m_sur_replay"] >= 0.0 and row["m_hf"] >= 0.0
            else "FS" if row["m_sur_replay"] >= 0.0
            else "CR" if row["m_hf"] >= 0.0
            else "UA"
        )
        row["replay_label_false_safe"] = int(row["replay_regime"] == "FS")
        row["development_source"] = source
        rescored.append(row)
    return rescored


def validate_fresh(rows: list[dict[str, Any]], expected_n: int = 100) -> None:
    expected = {(pde, model, i) for pde in PDES for model in MODELS for i in range(expected_n)}
    keys = [(str(row["pde"]), str(row["model"]), int(row["ic_idx"])) for row in rows]
    if len(rows) != len(expected) or len(set(keys)) != len(expected) or set(keys) != expected:
        raise RuntimeError(f"Fresh rows incomplete: rows={len(rows)} unique={len(set(keys))}")
    if {len(row["residual_features"]) for row in rows} != {19}:
        raise RuntimeError("Fresh residual dimension is not 19")
    if max(float(row["initial_state_max_abs_error_to_regenerated"]) for row in rows) > 1e-6:
        raise RuntimeError("Fresh replay initial-state mismatch")


def finite_qhat(values: list[float], coverage: float) -> tuple[float, dict[str, Any]]:
    qhat, rank, n, feasible = finite_sample_upper_quantile(
        np.asarray(values, dtype=np.float64), coverage
    )
    if not feasible or not math.isfinite(qhat):
        raise RuntimeError(f"Infeasible qhat: n={n} rank={rank}")
    return qhat, {"qhat": qhat, "rank": rank, "n": n, "coverage": coverage}


def pde_group_upper(
    rows: list[dict[str, Any]], prediction: np.ndarray, cal_idx: np.ndarray, test_idx: np.ndarray, coverage: float
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    gap = np.asarray([float(row["target_gap_hf_minus_replayed_sur"]) for row in rows], dtype=np.float64)
    score = gap - prediction
    upper = np.full(len(rows), np.nan, dtype=np.float64)
    records = []
    for pde in PDES:
        groups: dict[int, list[float]] = defaultdict(list)
        for index in cal_idx:
            if rows[index]["pde"] == pde:
                groups[int(rows[index]["ic_idx"])].append(float(score[index]))
        if len(groups) != 40 or any(len(values) != 4 for values in groups.values()):
            raise RuntimeError(f"Malformed calibration groups for {pde}")
        qhat, meta = finite_qhat([max(values) for values in groups.values()], coverage)
        local_test = np.asarray([index for index in test_idx if rows[index]["pde"] == pde], dtype=np.int64)
        upper[local_test] = prediction[local_test] + qhat
        records.append({"pde": pde, "unit": "IC max over four models", **meta})
    if np.any(~np.isfinite(upper[test_idx])):
        raise RuntimeError("Candidate upper bounds are incomplete")
    return upper, records


def cp_lower(successes: int, trials: int, confidence: float = 0.95) -> float:
    if successes <= 0:
        return 0.0
    return float(beta.ppf((1.0 - confidence) / 2.0, successes, trials - successes + 1))


def candidate_coverage(rows: list[dict[str, Any]], test_idx: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    covered = {
        (str(rows[index]["pde"]), int(rows[index]["ic_idx"])): [] for index in test_idx
    }
    for index in test_idx:
        key = (str(rows[index]["pde"]), int(rows[index]["ic_idx"]))
        covered[key].append(float(rows[index]["target_gap_hf_minus_replayed_sur"]) <= float(upper[index]))
    if any(len(values) != 4 for values in covered.values()):
        raise RuntimeError("Incomplete candidate test groups")
    group_values = [all(values) for values in covered.values()]
    success = int(sum(group_values))
    return {
        "group_n": len(group_values),
        "group_covered": success,
        "group_coverage": success / len(group_values),
        "group_coverage_cp95_lower": cp_lower(success, len(group_values)),
    }


def composed_metrics(
    rows: list[dict[str, Any]],
    test_idx: np.ndarray,
    accepted: np.ndarray,
    continuation: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    candidate_unsafe = 0
    continuation_executed = 0
    continuation_unsafe = 0
    continuation_uncertified = 0
    final_failure = 0
    for index in test_idx:
        row = rows[index]
        if accepted[index]:
            unsafe = float(row["m_hf"]) < 0.0
            candidate_unsafe += int(unsafe)
            final_failure += int(unsafe)
        else:
            record = continuation[(str(row["pde"]), int(row["ic_idx"]))]
            if int(record["continuation_certified"]):
                continuation_executed += 1
                unsafe = not bool(int(record["continuation_hf_safe"]))
                continuation_unsafe += int(unsafe)
                final_failure += int(unsafe)
            else:
                continuation_uncertified += 1
                final_failure += 1
    n = len(test_idx)
    accepted_n = int(np.sum(accepted[test_idx]))
    return {
        "n": n,
        "candidate_accepted": accepted_n,
        "candidate_rejected": n - accepted_n,
        "candidate_accepted_unsafe": candidate_unsafe,
        "continuation_executed": continuation_executed,
        "continuation_uncertified": continuation_uncertified,
        "continuation_executed_unsafe": continuation_unsafe,
        "executed_unsafe": candidate_unsafe + continuation_unsafe,
        "final_failure": final_failure,
        "final_failure_rate": final_failure / n,
    }


def scope_metrics(
    rows: list[dict[str, Any]], test_idx: np.ndarray, accepted: np.ndarray, continuation: dict[tuple[str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for pde in PDES:
        local = np.asarray([index for index in test_idx if rows[index]["pde"] == pde], dtype=np.int64)
        output.append({"pde": pde, **composed_metrics(rows, local, accepted, continuation)})
    return output


def continuation_test_metrics(
    continuation: dict[tuple[str, int], dict[str, Any]], assignments: dict[str, Any]
) -> dict[str, Any]:
    by_pde = {}
    for pde in PDES:
        indices = assignments[pde]["test_ic_idx"]
        local = [continuation[(pde, int(ic_idx))] for ic_idx in indices]
        by_pde[pde] = {
            "n": len(local),
            "upper_coverage": int(sum(int(row["upper_covered"]) for row in local)),
            "upper_coverage_rate": float(np.mean([int(row["upper_covered"]) for row in local])),
            "certified_domain": int(sum(int(row["continuation_certified"]) for row in local)),
            "certified_domain_rate": float(np.mean([int(row["continuation_certified"]) for row in local])),
            "certified_but_hf_unsafe": int(
                sum(int(row["continuation_certified"]) and not int(row["continuation_hf_safe"]) for row in local)
            ),
        }
    return by_pde


def run_self_test() -> None:
    rows = []
    for pde in PDES:
        for model in MODELS:
            for ic_idx in range(100):
                gap = 0.001 * ic_idx
                rows.append(
                    {
                        "pde": pde,
                        "model": model,
                        "pair": f"{pde}:{model}",
                        "ic_idx": ic_idx,
                        "target_gap_hf_minus_replayed_sur": gap,
                        "m_sur_replay": 0.2,
                        "m_hf": 0.2 - gap,
                    }
                )
    rows.sort(key=lambda row: (row["pde"], row["model"], row["ic_idx"]))
    cal_idx = np.asarray([i for i, row in enumerate(rows) if row["ic_idx"] < 40], dtype=np.int64)
    test_idx = np.asarray([i for i, row in enumerate(rows) if row["ic_idx"] >= 40], dtype=np.int64)
    prediction = np.zeros(len(rows), dtype=np.float64)
    upper, records = pde_group_upper(rows, prediction, cal_idx, test_idx, 0.95)
    assert len(records) == 3 and all(record["n"] == 40 for record in records)
    coverage = candidate_coverage(rows, test_idx, upper)
    assert coverage["group_n"] == 180
    accepted = np.zeros(len(rows), dtype=bool)
    accepted[test_idx] = True
    continuation = {
        (pde, ic_idx): {"continuation_certified": 1, "continuation_hf_safe": 1, "upper_covered": 1}
        for pde in PDES for ic_idx in range(100)
    }
    metrics = composed_metrics(rows, test_idx, accepted, continuation)
    assert metrics["candidate_accepted"] == 720
    assert metrics["final_failure"] == 0
    accepted[test_idx] = False
    continuation[("burgers", 40)]["continuation_certified"] = 0
    metrics = composed_metrics(rows, test_idx, accepted, continuation)
    assert metrics["continuation_uncertified"] == 4
    assert metrics["final_failure"] == 4
    print(json.dumps({"self_test": "PASS", "rows": len(rows), "test_rows": len(test_idx)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=BASE / "external_data" / "analysis" / "recoverability" / "protocol.json")
    parser.add_argument("--split", type=Path, default=BASE / "external_data" / "analysis" / "recoverability" / "state_split.json")
    parser.add_argument("--discovery-glob", default=str(BASE / "external_data" / "analysis" / "mismatch_detection" / "discovery_replay" / "shard_*.jsonl"))
    parser.add_argument("--development-glob", default=str(BASE / "external_data" / "analysis" / "mismatch_detection" / "detector_development" / "shard_*.jsonl"))
    parser.add_argument("--evaluation-glob", default=str(BASE / "external_data" / "analysis" / "mismatch_detection" / "evaluation" / "shard_*.jsonl"))
    parser.add_argument("--discovery-root", type=Path, default=BASE / "external_data/decision_archives/discovery")
    parser.add_argument("--development-root", type=Path, default=BASE / "external_data/decision_archives/detector_development")
    parser.add_argument("--evaluation-audit", type=Path, default=BASE / "external_data" / "analysis" / "recoverability" / "evaluation_audit.json")
    parser.add_argument("--continuations", type=Path, default=BASE / "external_data" / "analysis" / "recoverability" / "continuation_rows.jsonl")
    parser.add_argument("--outdir", type=Path, default=BASE / "outputs" / "continuation_evaluation")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    assignments = split["assignments"]
    scale = float(protocol["physical_boundary"]["scale"])
    candidate_coverage_target = 0.95

    discovery_rows = load_rows(args.discovery_glob)
    if len(discovery_rows) != 2400:
        raise RuntimeError(f"Expected 2,400 discovery rows, found {len(discovery_rows)}")
    development_rows = load_rows(args.development_glob)
    if len(development_rows) != 1200:
        raise RuntimeError(f"Expected 1200 detector development study rows, found {len(development_rows)}")
    development = rescore_development(
        discovery_rows, z0_map(args.discovery_root, 200), scale, "discovery"
    )
    development += rescore_development(development_rows, z0_map(args.development_root, 100), scale, "detector_development")
    fresh = load_rows(args.evaluation_glob)
    validate_fresh(fresh)
    fresh_audit = json.loads(args.evaluation_audit.read_text(encoding="utf-8"))
    continuation_rows = list(iter_jsonl(args.continuations))
    continuation = {(str(row["pde"]), int(row["ic_idx"])): row for row in continuation_rows}
    if len(continuation) != 300:
        raise RuntimeError(f"Expected 300 continuation rows, found {len(continuation)}")

    cal_idx = np.asarray(
        [
            i for i, row in enumerate(fresh)
            if int(row["ic_idx"]) in set(assignments[str(row["pde"])]["calibration_ic_idx"])
        ],
        dtype=np.int64,
    )
    test_idx = np.asarray(
        [
            i for i, row in enumerate(fresh)
            if int(row["ic_idx"]) in set(assignments[str(row["pde"])]["test_ic_idx"])
        ],
        dtype=np.int64,
    )
    if len(cal_idx) != 480 or len(test_idx) != 720:
        raise RuntimeError(f"Frozen split mismatch: cal={len(cal_idx)} test={len(test_idx)}")

    dev_gap = np.asarray([float(row["target_gap_hf_minus_replayed_sur"]) for row in development])
    residual_model = fit_gap_model(residual_feature_matrix(development), dev_gap, 42)
    predictions = {
        "residual_detector": np.asarray(
            residual_model.predict(residual_feature_matrix(fresh)), dtype=np.float64
        )
    }

    policies = {}
    accepted_by_policy = {}
    for method, prediction in predictions.items():
        upper, qhats = pde_group_upper(fresh, prediction, cal_idx, test_idx, candidate_coverage_target)
        accepted = np.zeros(len(fresh), dtype=bool)
        accepted[test_idx] = upper[test_idx] <= np.asarray([float(fresh[i]["m_sur_replay"]) for i in test_idx])
        accepted_by_policy[method] = accepted
        policies[method] = {
            "candidate_coverage": candidate_coverage(fresh, test_idx, upper),
            "candidate_qhats": qhats,
            "overall": composed_metrics(fresh, test_idx, accepted, continuation),
            "by_pde": scope_metrics(fresh, test_idx, accepted, continuation),
        }

    raw = np.zeros(len(fresh), dtype=bool)
    raw[test_idx] = np.asarray([float(fresh[index]["m_sur_replay"]) >= 0.0 for index in test_idx])
    hf_oracle = np.zeros(len(fresh), dtype=bool)
    hf_oracle[test_idx] = np.asarray([float(fresh[index]["m_hf"]) >= 0.0 for index in test_idx])
    policies["raw_surrogate"] = {
        "overall": composed_metrics(fresh, test_idx, raw, continuation),
        "by_pde": scope_metrics(fresh, test_idx, raw, continuation),
    }
    policies["hf_candidate_oracle"] = {
        "overall": composed_metrics(fresh, test_idx, hf_oracle, continuation),
        "by_pde": scope_metrics(fresh, test_idx, hf_oracle, continuation),
    }

    continuation_metrics = continuation_test_metrics(continuation, assignments)
    residual = policies["residual_detector"]["overall"]
    raw_metrics = policies["raw_surrogate"]["overall"]
    checks = {
        "common_boundary_audit": bool(fresh_audit["passed"]),
        "evaluation_contains_720_model_state_scenarios": len(test_idx) == 720,
        "evaluation_contains_180_physical_state_groups": len(test_idx) // 4 == 180,
        "zero_unsafe_continuation_inside_certified_domain": all(
            continuation_metrics[pde]["certified_but_hf_unsafe"] == 0 for pde in PDES
        ),
        "residual_detector_accepts_247_proposals": residual["candidate_accepted"] == 247,
        "residual_detector_has_zero_unsafe_executions": residual["executed_unsafe"] == 0,
        "residual_detector_has_48_unresolved_scenarios": residual["final_failure"] == 48,
        "raw_selection_has_17_unsafe_executions": raw_metrics["executed_unsafe"] == 17,
        "raw_selection_has_50_unresolved_scenarios": raw_metrics["final_failure"] == 50,
    }
    passed = all(checks.values())
    payload = {
        "protocol": "residual detector and continuation availability evaluation",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "protocol_sha256": sha256(args.protocol),
        "split_sha256": sha256(args.split),
        "development_rows": len(development),
        "fresh_rows": len(fresh),
        "fresh_regimes": dict(Counter(str(fresh[index]["replay_regime"]) for index in test_idx)),
        "continuation_test": continuation_metrics,
        "policies": policies,
        "success_checks": checks,
        "passed": passed,
        "verdict": "EVALUATION_PASS" if passed else "EVALUATION_FAIL",
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    result = args.outdir / "composed_confirmation.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    verdict = {
        "verdict": payload["verdict"],
        "passed": passed,
        "success_checks": checks,
        "residual_detector": residual,
        "raw": raw_metrics,
        "result_sha256": sha256(result),
    }
    (args.outdir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

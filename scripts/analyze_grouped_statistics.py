"""Grouped inference and operating-point sensitivity analyses.

This script does not rerun any PDE trajectory. It treats the original discovery decision study
artifacts as the canonical discovery record, resamples physical states rather
than model--condition rows, and reconstructs the frozen 720-scenario detector
test to expose its safety--acceptance trade-off.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import evaluate_continuation_availability as g15
from mismatch_detection import fit_gap_model, residual_feature_matrix
from audit_data_integrity import (
    BASE,
    MODELS,
    PDES,
    canonical_rows,
    read_jsonl,
    regime,
    write_jsonl,
)


REGIMES = (
    "safe_agreement",
    "false_safe",
    "conservative_rejection",
    "unsafe_agreement",
)


def percentile_interval(values: Iterable[float]) -> list[float]:
    values = np.asarray(list(values), dtype=np.float64)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def load_discovery(cache_root: Path) -> list[dict[str, Any]]:
    rows, _ = canonical_rows(cache_root)
    by_key = {(row["pde"], row["model"], row["ic_idx"]): row for row in rows}
    for pde in PDES:
        for model in MODELS:
            path = cache_root / pde / model / "challenge_conditions" / "decision_data.npz"
            with np.load(path, allow_pickle=True) as data:
                z0 = np.asarray(data["z0"], dtype=np.float64)
                tags = np.asarray(data["tags"]).astype(str)
            for ic_idx in range(len(z0)):
                row = by_key[(pde, model, ic_idx)]
                row["z0"] = float(z0[ic_idx])
                row["tag"] = str(tags[ic_idx])
                row["group"] = f"{pde}:ic{ic_idx:03d}"
    return rows


def threshold_maps(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    output: dict[str, dict[str, float]] = {
        "original_pair_specific": {},
        "statewise_median_original": {},
        "statewise_minimum_original": {},
        "statewise_maximum_original": {},
        "initial_risk_x1.02": {},
        "initial_risk_x1.20": {},
    }
    for group, local in groups.items():
        limits = np.asarray([row["z_limit"] for row in local], dtype=np.float64)
        z0 = float(np.median([row["z0"] for row in local]))
        output["statewise_median_original"][group] = float(np.median(limits))
        output["statewise_minimum_original"][group] = float(np.min(limits))
        output["statewise_maximum_original"][group] = float(np.max(limits))
        output["initial_risk_x1.02"][group] = 1.02 * z0
        output["initial_risk_x1.20"][group] = 1.20 * z0
    return output


def labels_for_threshold(
    rows: list[dict[str, Any]], name: str, mapping: dict[str, float]
) -> list[str]:
    labels = []
    for row in rows:
        limit = row["z_limit"] if name == "original_pair_specific" else mapping[row["group"]]
        labels.append(regime(limit - row["z_sur"], limit - row["z_hf"]))
    return labels


def summarize_labels(labels: list[str]) -> dict[str, Any]:
    counts = Counter(labels)
    accepted = counts["safe_agreement"] + counts["false_safe"]
    return {
        "n": len(labels),
        "counts": {name: int(counts[name]) for name in REGIMES},
        "false_safe_fraction_all": counts["false_safe"] / len(labels),
        "false_safe_fraction_accepted": counts["false_safe"] / accepted if accepted else None,
        "accepted_fraction": accepted / len(labels),
    }


def grouped_discovery_bootstrap(
    rows: list[dict[str, Any]], labels: list[str], n_bootstrap: int, seed: int
) -> dict[str, Any]:
    by_group: dict[str, list[int]] = defaultdict(list)
    groups_by_pde: dict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[row["group"]].append(index)
    for group in by_group:
        groups_by_pde[group.split(":", 1)[0]].append(group)
    if any(len(indices) != 4 for indices in by_group.values()):
        raise RuntimeError("Discovery groups must contain four surrogate evaluations")
    point = summarize_labels(labels)
    point["physical_state_groups"] = len(by_group)
    point["states_with_at_least_one_false_safe"] = int(
        sum(any(labels[index] == "false_safe" for index in indices) for indices in by_group.values())
    )
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_bootstrap):
        sampled_indices: list[int] = []
        sampled_state_events: list[float] = []
        for pde in PDES:
            candidates = groups_by_pde[pde]
            selected = rng.choice(candidates, size=len(candidates), replace=True)
            for group in selected:
                indices = by_group[str(group)]
                sampled_indices.extend(indices)
                sampled_state_events.append(float(any(labels[i] == "false_safe" for i in indices)))
        local = summarize_labels([labels[index] for index in sampled_indices])
        for key in ("false_safe_fraction_all", "false_safe_fraction_accepted", "accepted_fraction"):
            draws[key].append(local[key])
        draws["state_false_safe_fraction"].append(float(np.mean(sampled_state_events)))
    point["cluster_bootstrap_95ci"] = {
        key: percentile_interval(values) for key, values in draws.items()
    }
    return point


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_critical_event_sensitivity_files(critical_event_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(critical_event_path)
    definitions = {
        "regime_stable": [row for row in rows if not int(row["fresh_to_canonical_regime_changed"])],
        "replay_error_at_most_1e-2": [
            row for row in rows if float(row["z_sur_abs_error_to_saved_replay"]) <= 1e-2
        ],
        "strict_stable": [
            row
            for row in rows
            if not int(row["fresh_to_canonical_regime_changed"])
            and float(row["z_sur_abs_error_to_saved_replay"]) <= 1e-2
        ],
    }
    output: dict[str, Any] = {}
    for name, local in definitions.items():
        path = output_dir / f"critical_event_{name}.jsonl"
        write_jsonl(path, local)
        output[name] = {
            "path": str(path.resolve()),
            "n": len(local),
            "false_safe": int(sum(row["regime"] == "false_safe" for row in local)),
            "accepted": int(sum(row["regime"] in {"false_safe", "safe_agreement"} for row in local)),
        }
    return output


def detector_group_records(
    rows: list[dict[str, Any]],
    test_idx: np.ndarray,
    accepted: np.ndarray,
    continuation: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in test_idx:
        groups[(str(rows[index]["pde"]), int(rows[index]["ic_idx"]))].append(int(index))
    records = []
    for (pde, ic_idx), indices in sorted(groups.items()):
        if len(indices) != 4:
            raise RuntimeError("Detector test groups must contain four surrogate evaluations")
        continuation_row = continuation[(pde, ic_idx)]
        accepted_count = 0
        unsafe_count = 0
        unresolved_count = 0
        for index in indices:
            if accepted[index]:
                accepted_count += 1
                unsafe = float(rows[index]["m_hf"]) < 0.0
                unsafe_count += int(unsafe)
                unresolved_count += int(unsafe)
            else:
                unavailable = not bool(int(continuation_row["continuation_certified"]))
                unsafe_continuation = bool(int(continuation_row["continuation_certified"])) and not bool(
                    int(continuation_row["continuation_hf_safe"])
                )
                unsafe_count += int(unsafe_continuation)
                unresolved_count += int(unavailable or unsafe_continuation)
        records.append(
            {
                "pde": pde,
                "ic_idx": ic_idx,
                "accepted": accepted_count,
                "unsafe": unsafe_count,
                "unresolved": unresolved_count,
                "n": 4,
            }
        )
    return records


def bootstrap_detector_groups(
    records: list[dict[str, Any]], n_bootstrap: int, seed: int
) -> dict[str, list[float]]:
    by_pde = {pde: [row for row in records if row["pde"] == pde] for pde in PDES}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_bootstrap):
        sample = []
        for pde in PDES:
            local = by_pde[pde]
            selected = rng.integers(0, len(local), size=len(local))
            sample.extend(local[index] for index in selected)
        n = sum(row["n"] for row in sample)
        draws["accepted_fraction"].append(sum(row["accepted"] for row in sample) / n)
        draws["unsafe_fraction"].append(sum(row["unsafe"] for row in sample) / n)
        draws["unresolved_fraction"].append(sum(row["unresolved"] for row in sample) / n)
    return {key: percentile_interval(values) for key, values in draws.items()}


def detector_sweep(n_bootstrap: int, seed: int) -> dict[str, Any]:
    protocol_path = BASE / "external_data/analysis/recoverability/protocol.json"
    split_path = BASE / "external_data/analysis/recoverability/state_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = split["assignments"]
    scale = float(json.loads(protocol_path.read_text(encoding="utf-8"))["physical_boundary"]["scale"])

    discovery = g15.load_rows(
        str(BASE / "external_data/analysis/mismatch_detection/discovery_replay/shard_*.jsonl")
    )
    detector_development = g15.load_rows(
        str(BASE / "external_data/analysis/mismatch_detection/detector_development/shard_*.jsonl")
    )
    development = g15.rescore_development(
        discovery,
        g15.z0_map(BASE / "external_data/decision_archives/discovery", 200),
        scale,
        "discovery",
    )
    development += g15.rescore_development(
        detector_development,
        g15.z0_map(BASE / "external_data/decision_archives/detector_development", 100),
        scale,
        "detector_development",
    )
    evaluation = g15.load_rows(
        str(BASE / "external_data/analysis/mismatch_detection/evaluation/shard_*.jsonl")
    )
    g15.validate_fresh(evaluation)
    continuation_rows = list(
        g15.iter_jsonl(BASE / "external_data/analysis/recoverability/continuation_rows.jsonl")
    )
    continuation = {(str(row["pde"]), int(row["ic_idx"])): row for row in continuation_rows}
    cal_idx = np.asarray(
        [
            i
            for i, row in enumerate(evaluation)
            if int(row["ic_idx"]) in set(assignments[str(row["pde"])]["calibration_ic_idx"])
        ],
        dtype=np.int64,
    )
    test_idx = np.asarray(
        [
            i
            for i, row in enumerate(evaluation)
            if int(row["ic_idx"]) in set(assignments[str(row["pde"])]["test_ic_idx"])
        ],
        dtype=np.int64,
    )
    target = np.asarray([float(row["target_gap_hf_minus_replayed_sur"]) for row in development])
    model = fit_gap_model(residual_feature_matrix(development), target, 42)
    prediction = np.asarray(
        model.predict(residual_feature_matrix(evaluation)), dtype=np.float64
    )

    rows_out = []
    for coverage in (0.80, 0.85, 0.90, 0.925, 0.95, 0.975):
        upper, qhats = g15.pde_group_upper(evaluation, prediction, cal_idx, test_idx, coverage)
        accepted = np.zeros(len(evaluation), dtype=bool)
        margins = np.asarray([float(row["m_sur_replay"]) for row in evaluation], dtype=np.float64)
        accepted[test_idx] = upper[test_idx] <= margins[test_idx]
        metrics = g15.composed_metrics(evaluation, test_idx, accepted, continuation)
        group_records = detector_group_records(evaluation, test_idx, accepted, continuation)
        rows_out.append(
            {
                "coverage_nominal": coverage,
                "coverage_group_empirical": g15.candidate_coverage(evaluation, test_idx, upper)["group_coverage"],
                "accepted": metrics["candidate_accepted"],
                "accepted_fraction": metrics["candidate_accepted"] / metrics["n"],
                "executed_unsafe": metrics["executed_unsafe"],
                "unresolved": metrics["final_failure"],
                "unresolved_fraction": metrics["final_failure_rate"],
                "qhat_by_pde": {row["pde"]: row["qhat"] for row in qhats},
                "cluster_bootstrap_95ci": bootstrap_detector_groups(
                    group_records, n_bootstrap, seed + int(coverage * 1000)
                ),
            }
        )

    baselines = {}
    for name, acceptance_rule in {
        "raw_surrogate": lambda row: float(row["m_sur_replay"]) >= 0.0,
        "hf_oracle": lambda row: float(row["m_hf"]) >= 0.0,
    }.items():
        accepted = np.zeros(len(evaluation), dtype=bool)
        accepted[test_idx] = [acceptance_rule(evaluation[index]) for index in test_idx]
        metrics = g15.composed_metrics(evaluation, test_idx, accepted, continuation)
        records = detector_group_records(evaluation, test_idx, accepted, continuation)
        baselines[name] = {
            **metrics,
            "cluster_bootstrap_95ci": bootstrap_detector_groups(records, n_bootstrap, seed + 9000),
        }
    return {"coverage_sweep": rows_out, "baselines": baselines}


def write_report(path: Path, payload: dict[str, Any]) -> None:
    original = payload["threshold_sensitivity"]["original_pair_specific"]
    lines = [
        "# Grouped statistical analysis",
        "",
        "## Canonical challenge cohort",
        "",
        f"- False-safe model--condition evaluations: {original['counts']['false_safe']} / {original['n']}",
        f"- State-grouped 95% CI for the pooled false-safe fraction: {original['cluster_bootstrap_95ci']['false_safe_fraction_all']}",
        f"- Physical state groups with at least one false-safe model: {original['states_with_at_least_one_false_safe']} / {original['physical_state_groups']}",
        "- These rates describe a deliberately selected challenge cohort, not an operating-population prevalence.",
        "",
        "## Threshold sensitivity",
        "",
        "| Definition | False safe | Accepted | FS/all | FS/accepted |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in payload["threshold_sensitivity"].items():
        accepted = result["counts"]["safe_agreement"] + result["counts"]["false_safe"]
        lines.append(
            f"| {name} | {result['counts']['false_safe']} | {accepted} | "
            f"{result['false_safe_fraction_all']:.4f} | {result['false_safe_fraction_accepted']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Threshold variants change the scientific question as well as the count. The original "
            "limits measure reversals under the deployed pair-specific decision rule; common limits "
            "are sensitivity analyses and must not be presented as equivalent safety definitions.",
            "",
            "## Residual detector operating points",
            "",
            "| Nominal coverage | Accepted | Unsafe | Unresolved |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in payload["detector"]["coverage_sweep"]:
        lines.append(
            f"| {row['coverage_nominal']:.3f} | {row['accepted']}/720 | "
            f"{row['executed_unsafe']}/720 | {row['unresolved']}/720 |"
        )
    lines.extend(
        [
            "",
            "The detector should therefore be reported as a safety--acceptance operating curve. "
            "The 95% point removed observed unsafe executions by accepting fewer proposals; it did "
            "not create additional physically feasible continuations.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=BASE / "external_data/decision_archives/discovery")
    parser.add_argument(
        "--canonical-critical-events",
        type=Path,
        default=BASE / "external_data/analysis/critical_events/critical_event_features.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "outputs" / "audits" / "grouped_statistics",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_discovery(args.cache_root)
    mappings = threshold_maps(rows)
    threshold_results: dict[str, Any] = {}
    threshold_csv = []
    for offset, (name, mapping) in enumerate(mappings.items()):
        labels = labels_for_threshold(rows, name, mapping)
        result = grouped_discovery_bootstrap(rows, labels, args.bootstrap, args.seed + offset)
        threshold_results[name] = result
        threshold_csv.append(
            {
                "threshold_definition": name,
                **result["counts"],
                "false_safe_fraction_all": result["false_safe_fraction_all"],
                "false_safe_fraction_accepted": result["false_safe_fraction_accepted"],
                "accepted_fraction": result["accepted_fraction"],
                "states_with_false_safe": result["states_with_at_least_one_false_safe"],
            }
        )

    payload = {
        "analysis": "Grouped statistical analysis",
        "bootstrap_replicates": args.bootstrap,
        "resampling_unit": "physical state; four surrogate evaluations remain grouped",
        "threshold_sensitivity": threshold_results,
        "critical_event_replay_sensitivity_files": build_critical_event_sensitivity_files(args.canonical_critical_events, args.output_dir),
        "detector": detector_sweep(args.bootstrap, args.seed),
    }
    (args.output_dir / "statistical_reanalysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "threshold_sensitivity.csv", threshold_csv)
    write_csv(
        args.output_dir / "detector_operating_curve.csv",
        [
            {key: value for key, value in row.items() if key not in {"qhat_by_pde", "cluster_bootstrap_95ci"}}
            for row in payload["detector"]["coverage_sweep"]
        ],
    )
    write_report(args.output_dir / "STATISTICAL_REANALYSIS.md", payload)
    print(json.dumps({
        "output": str((args.output_dir / 'statistical_reanalysis.json').resolve()),
        "original": threshold_results["original_pair_specific"],
        "detector_curve": payload["detector"]["coverage_sweep"],
    }, indent=2))


if __name__ == "__main__":
    main()

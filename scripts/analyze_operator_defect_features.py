"""Evaluate PDE operator-defect features for false-safe decision detection.

This entry point reproduces the feature audit summarized in Extended Data
Fig. 4. All covariates are available from the surrogate rollout, proposed
control and known PDE operator. High-fidelity risks are used only as offline
labels for grouped cross-validation and conformal calibration.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from operator_defect_features import RESIDUAL_FEATURE_NAMES


BASE = Path(__file__).resolve().parents[1]
FEATURE_SETS = (
    "compact4",
    "residual19_only",
    "compact4_residual19",
    "phi6",
    "phi6_residual19",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def load_analysis_rows(
    risk_geometry_path: Path,
    operator_defect_path: Path,
    decision_archive_root: Path,
) -> list[dict[str, Any]]:
    risk_rows = {
        (row["pde"], row["model"], int(row["ic_idx"])): row
        for row in read_jsonl(risk_geometry_path)
    }
    defect_rows = {
        (row["pde"], row["model"], int(row["ic_idx"])): row
        for row in read_jsonl(operator_defect_path)
    }

    merged: list[dict[str, Any]] = []
    for archive_path in sorted(decision_archive_root.glob("*/*/challenge_conditions/decision_data.npz")):
        pde = archive_path.parts[-4]
        model = archive_path.parts[-3]
        with np.load(archive_path, allow_pickle=False) as archive:
            z_sur_all = np.asarray(archive["z_sur"], dtype=np.float64)
            z_hf_all = np.asarray(archive["z_hf"], dtype=np.float64)
            m_sur_all = np.asarray(archive["margin_sur"], dtype=np.float64)
            m_hf_all = np.asarray(archive["margin_hf"], dtype=np.float64)
            traces = np.asarray(archive["z_sur_ts_all"], dtype=np.float64)
        for index, (z_sur, z_hf, m_sur, m_hf, trace) in enumerate(
            zip(z_sur_all, z_hf_all, m_sur_all, m_hf_all, traces)
        ):
            key = (pde, model, index)
            if key not in risk_rows or key not in defect_rows:
                raise KeyError(f"Missing feature row for {key}")
            risk = risk_rows[key]
            residual = defect_rows[key]
            trace = np.asarray(trace, dtype=np.float64).ravel()
            increments = np.diff(trace)
            kappa = finite(risk.get("kappa_fd_full_est"))
            merged.append(
                {
                    "pde": pde,
                    "model": model,
                    "ic_idx": index,
                    "physical_group": f"{pde}:{index}",
                    "fold_id": index % 5,
                    "z_sur": float(z_sur),
                    "z_hf": float(z_hf),
                    "m_sur": float(m_sur),
                    "m_hf": float(m_hf),
                    "positive_gap": max(float(z_hf - z_sur), 0.0),
                    "false_safe": int(m_sur >= 0.0 and m_hf < 0.0),
                    "kappa_fd": kappa,
                    "log1p_kappa_fd": float(np.log1p(max(kappa, 0.0))),
                    "risk_accel": float(np.mean(np.abs(increments))) if len(increments) else 0.0,
                    "z_sur_std": float(np.std(trace)) if len(trace) else 0.0,
                    "z_sur_range": float(np.ptp(trace)) if len(trace) else 0.0,
                    "residual_features": [finite(value) for value in residual["residual_features"]],
                }
            )
    if len(merged) != len(risk_rows) or len(merged) != len(defect_rows):
        raise RuntimeError(
            "Expected one archived decision for each feature row; found "
            f"{len(merged)} decisions, {len(risk_rows)} risk rows and {len(defect_rows)} defect rows"
        )
    return sorted(merged, key=lambda row: (row["pde"], row["model"], row["ic_idx"]))


def feature_matrix(
    rows: list[dict[str, Any]], feature_set: str
) -> tuple[np.ndarray, list[str]]:
    compact = ["z_sur", "log1p_kappa_fd", "risk_accel", "z_sur_std"]
    expanded = [
        "z_sur",
        "m_sur",
        "risk_accel",
        "z_sur_range",
        "log1p_kappa_fd",
        "kappa_fd",
    ]
    if feature_set == "compact4":
        names = compact
    elif feature_set == "residual19_only":
        names = list(RESIDUAL_FEATURE_NAMES)
    elif feature_set == "compact4_residual19":
        names = compact + list(RESIDUAL_FEATURE_NAMES)
    elif feature_set == "phi6":
        names = expanded
    elif feature_set == "phi6_residual19":
        names = expanded + list(RESIDUAL_FEATURE_NAMES)
    else:
        raise KeyError(feature_set)

    values: list[list[float]] = []
    for row in rows:
        residual = dict(zip(RESIDUAL_FEATURE_NAMES, row["residual_features"]))
        values.append([finite(row.get(name, residual.get(name, 0.0))) for name in names])
    matrix = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6
    )
    return matrix, names


def out_of_fold_scores(
    features: np.ndarray, target: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    scores = np.zeros(len(target), dtype=np.float64)
    for fold in np.unique(folds):
        test = folds == fold
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.90,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=5,
            random_state=100 + int(fold),
        )
        model.fit(features[~test], target[~test])
        scores[test] = model.predict(features[test])
    return scores


def bootstrap_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    replicates: int,
    seed: int = 7,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        indices = generator.integers(0, len(labels), size=len(labels))
        sampled = labels[indices]
        if len(np.unique(sampled)) < 2:
            continue
        values.append(float(metric(sampled, scores[indices])))
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def ranking_metrics(
    rows: list[dict[str, Any]], bootstrap_replicates: int
) -> list[dict[str, Any]]:
    labels = np.asarray([row["false_safe"] for row in rows], dtype=np.int32)
    target = np.asarray([row["positive_gap"] for row in rows], dtype=np.float64)
    folds = np.asarray([row["fold_id"] for row in rows], dtype=np.int32)
    accepted = np.asarray([row["m_sur"] >= 0.0 for row in rows], dtype=bool)
    output: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        features, names = feature_matrix(rows, feature_set)
        scores = out_of_fold_scores(features, target, folds)
        for scope, mask in (("accepted_only", accepted), ("all_decisions", np.ones(len(rows), bool))):
            y, s = labels[mask], scores[mask]
            auroc = float(roc_auc_score(y, s))
            average_precision = float(average_precision_score(y, s))
            auroc_low, auroc_high = bootstrap_interval(
                y, s, roc_auc_score, bootstrap_replicates
            )
            ap_low, ap_high = bootstrap_interval(
                y, s, average_precision_score, bootstrap_replicates
            )
            output.append(
                {
                    "feature_set": feature_set,
                    "scope": scope,
                    "n": int(mask.sum()),
                    "n_false_safe": int(y.sum()),
                    "n_features": len(names),
                    "feature_names": ";".join(names),
                    "auroc": auroc,
                    "auroc_ci95_low": auroc_low,
                    "auroc_ci95_high": auroc_high,
                    "average_precision": average_precision,
                    "average_precision_ci95_low": ap_low,
                    "average_precision_ci95_high": ap_high,
                }
            )
    return output


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    rank = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    if not len(scores) or rank > len(scores):
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def global_cqr_bounds(
    features: np.ndarray, target: np.ndarray, folds: np.ndarray, alpha: float
) -> np.ndarray:
    bounds = np.zeros(len(target), dtype=np.float64)
    unique_folds = list(np.unique(folds))
    for position, fold in enumerate(unique_folds):
        test = folds == fold
        calibrate = folds == unique_folds[(position + 1) % len(unique_folds)]
        train = ~(test | calibrate)
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=1.0 - alpha,
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=5,
            random_state=0,
        )
        model.fit(features[train], target[train])
        correction = conformal_quantile(
            target[calibrate] - model.predict(features[calibrate]), alpha
        )
        bounds[test] = model.predict(features[test]) + correction
    return bounds


def operating_metrics(rows: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    target = np.asarray([row["positive_gap"] for row in rows], dtype=np.float64)
    m_sur = np.asarray([row["m_sur"] for row in rows], dtype=np.float64)
    m_hf = np.asarray([row["m_hf"] for row in rows], dtype=np.float64)
    folds = np.asarray([row["fold_id"] for row in rows], dtype=np.int32)
    output: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        features, names = feature_matrix(rows, feature_set)
        bounds = global_cqr_bounds(features, target, folds, alpha)
        accept = m_sur - np.maximum(bounds, 0.0) >= 0.0
        false_safe = (m_sur >= 0.0) & (m_hf < 0.0)
        output.append(
            {
                "feature_set": feature_set,
                "alpha": alpha,
                "n_features": len(names),
                "accepted_unsafe": int(np.sum(accept & (m_hf < 0.0))),
                "rejected_decisions": int(np.sum(~accept)),
                "safe_rejections": int(np.sum((~accept) & (m_hf >= 0.0))),
                "false_safe_blocked": int(np.sum(false_safe & (~accept))),
                "false_safe_total": int(np.sum(false_safe)),
                "empirical_coverage": float(np.mean(target <= bounds)),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--risk-geometry",
        type=Path,
        default=BASE / "external_data" / "analysis" / "risk_geometry" / "decision_features.jsonl",
    )
    parser.add_argument(
        "--operator-defects",
        type=Path,
        default=BASE
        / "external_data"
        / "analysis"
        / "operator_defect_features"
        / "decision_features.jsonl",
    )
    parser.add_argument(
        "--decision-archive-root",
        type=Path,
        default=BASE / "external_data" / "decision_archives" / "discovery",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=BASE / "outputs" / "operator_defect_feature_audit",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    rows = load_analysis_rows(
        args.risk_geometry, args.operator_defects, args.decision_archive_root
    )
    accepted = sum(row["m_sur"] >= 0.0 for row in rows)
    false_safe = sum(row["false_safe"] for row in rows)
    if (len(rows), accepted, false_safe) != (2400, 1791, 116):
        raise RuntimeError(
            "Discovery archive does not match the frozen feature audit: "
            f"rows={len(rows)}, accepted={accepted}, false_safe={false_safe}"
        )

    ranking = ranking_metrics(rows, args.bootstrap_replicates)
    operating = operating_metrics(rows, args.alpha)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_directory / "ranking_metrics.csv", ranking)
    write_csv(args.output_directory / "conformal_operating_metrics.csv", operating)
    summary = {
        "analysis": "PDE operator-defect feature audit",
        "n_decisions": len(rows),
        "n_surrogate_accepted": accepted,
        "n_false_safe": false_safe,
        "grouping": "physical state; the four surrogate evaluations share a fold",
        "ranking_metrics": ranking,
        "conformal_operating_metrics": operating,
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

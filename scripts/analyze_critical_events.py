"""Analyze fidelity at decision-defining critical events."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parents[1]
MATCH_COVARIATES = ("log_field_rel_l2", "m_sur_norm")
CATEGORICAL_FEATURES = ("pair", "tag")
BASE_FEATURES = ("log_field_rel_l2", "m_sur_norm", "log_risk_trace_mae")
TAIL_UNSIGNED_FEATURES = BASE_FEATURES + (
    "log_tail_concentration_q99",
    "critical_abs_error_norm",
    "peak_time_displacement_frac",
    "peak_space_displacement_frac",
)
TAIL_SIGNED_FEATURES = TAIL_UNSIGNED_FEATURES + ("critical_signed_underestimate_norm",)
MODEL_FEATURE_SETS = {
    "aggregate": BASE_FEATURES,
    "extreme_unsigned": TAIL_UNSIGNED_FEATURES,
    "extreme_signed": TAIL_SIGNED_FEATURES,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def prepare_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    required = {
        "pde",
        "model",
        "ic_idx",
        "physical_group",
        "tag",
        "m_sur",
        "m_hf",
        "m_sur_norm",
        "field_rel_l2_all",
        "risk_trace_mae_norm",
        "risk_density_concentration_q99",
        "critical_signed_underestimate_norm",
        "critical_abs_error_norm",
        "peak_time_displacement_frac",
        "peak_space_displacement_frac",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing critical-event columns: {missing}")
    frame["pair"] = frame["pde"].astype(str) + ":" + frame["model"].astype(str)
    frame["surrogate_accepted"] = frame["m_sur"].astype(float) >= 0.0
    frame["false_safe"] = frame["surrogate_accepted"] & (frame["m_hf"].astype(float) < 0.0)
    frame["safe_agreement"] = frame["surrogate_accepted"] & (frame["m_hf"].astype(float) >= 0.0)
    frame["log_field_rel_l2"] = np.log10(np.maximum(frame["field_rel_l2_all"].astype(float), 1e-12))
    frame["log_risk_trace_mae"] = np.log10(
        np.maximum(frame["risk_trace_mae_norm"].astype(float), 1e-12)
    )
    frame["log_tail_concentration_q99"] = np.log(
        np.maximum(frame["risk_density_concentration_q99"].astype(float), 1e-12)
    )
    if not np.all(np.isfinite(frame[list(MATCH_COVARIATES)].to_numpy(dtype=float))):
        raise ValueError("Non-finite matching covariates")
    unique_columns = ["decision_id"] if "decision_id" in frame else ["pde", "model", "physical_group"]
    if frame.duplicated(unique_columns).any():
        raise ValueError("Duplicate critical-event model-condition keys")
    return frame


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centre = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (values - centre) / scale


def optimal_matches(
    accepted: pd.DataFrame, caliper: float
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    match_id = 0
    for (pair, tag), stratum in accepted.groupby(["pair", "tag"], sort=True):
        false_safe = stratum[stratum["false_safe"]].copy()
        controls = stratum[stratum["safe_agreement"]].copy()
        if false_safe.empty or controls.empty:
            continue
        combined = pd.concat([false_safe, controls], axis=0)
        standardized = standardize(combined[list(MATCH_COVARIATES)].to_numpy(dtype=float))
        n_false = len(false_safe)
        false_x = standardized[:n_false]
        control_x = standardized[n_false:]
        distances = np.sqrt(np.sum((false_x[:, None, :] - control_x[None, :, :]) ** 2, axis=2))
        max_coordinate_gap = np.max(
            np.abs(false_x[:, None, :] - control_x[None, :, :]), axis=2
        )
        penalized = distances.copy()
        penalized[max_coordinate_gap > float(caliper)] = 1e6
        row_ids, col_ids = linear_sum_assignment(penalized)
        for row_id, col_id in zip(row_ids, col_ids):
            if max_coordinate_gap[row_id, col_id] > float(caliper):
                continue
            fs = false_safe.iloc[int(row_id)]
            control = controls.iloc[int(col_id)]
            records.append(
                {
                    "match_id": match_id,
                    "pair": pair,
                    "tag": tag,
                    "distance": float(distances[row_id, col_id]),
                    "fs_row_index": int(fs.name),
                    "control_row_index": int(control.name),
                    "fs_physical_group": str(fs["physical_group"]),
                    "control_physical_group": str(control["physical_group"]),
                }
            )
            match_id += 1
    return pd.DataFrame(records)


def standardized_mean_difference(x1: Iterable[float], x0: Iterable[float]) -> float:
    x1 = np.asarray(list(x1), dtype=np.float64)
    x0 = np.asarray(list(x0), dtype=np.float64)
    pooled = math.sqrt(max(0.5 * (float(np.var(x1, ddof=1)) + float(np.var(x0, ddof=1))), 1e-30))
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def matched_metric_summary(
    frame: pd.DataFrame, matches: pd.DataFrame, metric: str
) -> dict[str, Any]:
    if matches.empty:
        return {"metric": metric, "n_matches": 0}
    fs_values = frame.loc[matches["fs_row_index"], metric].to_numpy(dtype=float)
    control_values = frame.loc[matches["control_row_index"], metric].to_numpy(dtype=float)
    differences = fs_values - control_values
    try:
        test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        p_value = float(test.pvalue)
        statistic = float(test.statistic)
    except ValueError:
        p_value = 1.0
        statistic = 0.0
    return {
        "metric": metric,
        "n_matches": int(len(matches)),
        "false_safe_mean": float(np.mean(fs_values)),
        "control_mean": float(np.mean(control_values)),
        "mean_paired_difference": float(np.mean(differences)),
        "median_paired_difference": float(np.median(differences)),
        "probability_difference_positive": float(np.mean(differences > 0.0)),
        "wilcoxon_statistic": statistic,
        "wilcoxon_two_sided_p": p_value,
    }


def cluster_bootstrap_matched_effects(
    accepted: pd.DataFrame,
    metrics: tuple[str, ...],
    caliper: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    groups = accepted["physical_group"].astype(str).unique()
    group_values = accepted["physical_group"].astype(str).to_numpy()
    by_group = {group: np.flatnonzero(group_values == group) for group in groups}
    estimates: dict[str, list[float]] = {metric: [] for metric in metrics}
    match_counts = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[str(group)] for group in sampled])
        bootstrap_frame = accepted.iloc[indices].reset_index(drop=True)
        matches = optimal_matches(bootstrap_frame, caliper)
        if matches.empty:
            continue
        for metric in metrics:
            fs_values = bootstrap_frame.loc[matches["fs_row_index"], metric].to_numpy(dtype=float)
            control_values = bootstrap_frame.loc[matches["control_row_index"], metric].to_numpy(dtype=float)
            estimates[metric].append(float(np.mean(fs_values - control_values)))
        match_counts.append(int(len(matches)))
    output = {}
    for metric, values in estimates.items():
        if not values:
            output[metric] = {"n_successful_bootstrap": 0}
            continue
        output[metric] = {
            "n_requested_bootstrap": int(n_bootstrap),
            "n_successful_bootstrap": int(len(values)),
            "mean_effect": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "median_bootstrap_matches": float(np.median(match_counts)),
        }
    return output


def build_pipeline(continuous_features: tuple[str, ...]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("continuous", StandardScaler(), list(continuous_features)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", transformer),
            (
                "classifier",
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, random_state=0),
            ),
        ]
    )


def prediction_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0 - 1e-8)
    return {
        "n": int(len(labels)),
        "positives": int(np.sum(labels)),
        "prevalence": float(np.mean(labels)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def grouped_oof_predictions(
    accepted: pd.DataFrame, n_splits: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = accepted["false_safe"].astype(int).to_numpy()
    groups = accepted["physical_group"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    predictions = pd.DataFrame(
        {
            "row_index": accepted.index.to_numpy(),
            "physical_group": groups,
            "pde": accepted["pde"].astype(str).to_numpy(),
            "model": accepted["model"].astype(str).to_numpy(),
            "label": labels,
        }
    )
    for name in MODEL_FEATURE_SETS:
        predictions[name] = np.nan
    fold_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(accepted, labels, groups)):
        train = accepted.iloc[train_idx]
        test = accepted.iloc[test_idx]
        for name, continuous in MODEL_FEATURE_SETS.items():
            model = build_pipeline(continuous)
            model.fit(train, train["false_safe"].astype(int))
            probability = model.predict_proba(test)[:, 1]
            predictions.loc[test_idx, name] = probability
            fold_rows.append(
                {
                    "fold": int(fold_id),
                    "model": name,
                    **prediction_metrics(test["false_safe"].astype(int).to_numpy(), probability),
                }
            )
    if predictions[list(MODEL_FEATURE_SETS)].isna().any().any():
        raise RuntimeError("Incomplete grouped out-of-fold predictions")
    summary = {
        name: prediction_metrics(labels, predictions[name].to_numpy(dtype=float))
        for name in MODEL_FEATURE_SETS
    }
    return predictions, {"overall": summary, "folds": fold_rows}


def repeated_grouped_oof_metrics(
    accepted: pd.DataFrame, n_splits: int, n_repeats: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(int(n_repeats)):
        repeat_seed = int(seed) + 101 * repeat
        _, summary = grouped_oof_predictions(accepted, int(n_splits), repeat_seed)
        for name, metrics in summary["overall"].items():
            rows.append(
                {
                    "repeat": int(repeat),
                    "seed": repeat_seed,
                    "model": name,
                    **metrics,
                }
            )
    table = pd.DataFrame(rows)
    metric_names = ("log_loss", "brier", "auroc", "average_precision")
    model_summary: dict[str, Any] = {}
    for name, subset in table.groupby("model", sort=False):
        model_summary[str(name)] = {
            metric: {
                "mean": float(subset[metric].mean()),
                "standard_deviation": float(subset[metric].std(ddof=1)),
                "minimum": float(subset[metric].min()),
                "maximum": float(subset[metric].max()),
            }
            for metric in metric_names
        }
    delta_summary: dict[str, Any] = {}
    baseline = table[table["model"] == "aggregate"].set_index("repeat")
    for name in ("extreme_unsigned", "extreme_signed"):
        extended = table[table["model"] == name].set_index("repeat")
        deltas = pd.DataFrame(
            {
                "log_loss_reduction": baseline["log_loss"] - extended["log_loss"],
                "brier_reduction": baseline["brier"] - extended["brier"],
                "auroc_gain": extended["auroc"] - baseline["auroc"],
                "average_precision_gain": extended["average_precision"]
                - baseline["average_precision"],
            }
        )
        delta_summary[name] = {
            metric: {
                "mean": float(deltas[metric].mean()),
                "standard_deviation": float(deltas[metric].std(ddof=1)),
                "minimum": float(deltas[metric].min()),
                "maximum": float(deltas[metric].max()),
                "fraction_positive": float((deltas[metric] > 0.0).mean()),
            }
            for metric in deltas.columns
        }
    return table, {
        "n_repeats": int(n_repeats),
        "n_splits": int(n_splits),
        "models": model_summary,
        "deltas_from_aggregate": delta_summary,
    }


def bootstrap_metric_deltas(
    predictions: pd.DataFrame, comparison: tuple[str, str], n_bootstrap: int, seed: int
) -> dict[str, Any]:
    baseline, extended = comparison
    groups = predictions["physical_group"].astype(str).unique()
    by_group = {
        group: predictions.index[predictions["physical_group"].astype(str) == group].to_numpy()
        for group in groups
    }
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[str(group)] for group in sampled])
        labels = predictions.loc[indices, "label"].to_numpy(dtype=int)
        if len(np.unique(labels)) < 2:
            continue
        base_metrics = prediction_metrics(labels, predictions.loc[indices, baseline].to_numpy(dtype=float))
        ext_metrics = prediction_metrics(labels, predictions.loc[indices, extended].to_numpy(dtype=float))
        records.append(
            {
                "log_loss_reduction": base_metrics["log_loss"] - ext_metrics["log_loss"],
                "brier_reduction": base_metrics["brier"] - ext_metrics["brier"],
                "auroc_gain": ext_metrics["auroc"] - base_metrics["auroc"],
                "average_precision_gain": ext_metrics["average_precision"]
                - base_metrics["average_precision"],
            }
        )
    output: dict[str, Any] = {
        "baseline": baseline,
        "extended": extended,
        "n_successful_bootstrap": len(records),
    }
    for metric in records[0] if records else []:
        values = np.asarray([row[metric] for row in records], dtype=float)
        output[metric] = {
            "mean": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
    return output


def leave_one_pde_out(accepted: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    if accepted["pde"].nunique() < 2:
        return records
    for held_out in sorted(accepted["pde"].unique()):
        train = accepted[accepted["pde"] != held_out]
        test = accepted[accepted["pde"] == held_out]
        if train.empty or train["false_safe"].nunique() < 2 or test["false_safe"].nunique() < 2:
            continue
        labels = test["false_safe"].astype(int).to_numpy()
        for name, continuous in MODEL_FEATURE_SETS.items():
            model = build_pipeline(continuous)
            model.fit(train, train["false_safe"].astype(int))
            probability = model.predict_proba(test)[:, 1]
            records.append(
                {
                    "held_out_pde": str(held_out),
                    "model": name,
                    **prediction_metrics(labels, probability),
                }
            )
    return records


def count_summary(frame: pd.DataFrame) -> dict[str, Any]:
    def one(subset: pd.DataFrame) -> dict[str, Any]:
        accepted = subset[subset["surrogate_accepted"]]
        return {
            "n": int(len(subset)),
            "surrogate_accepted": int(len(accepted)),
            "false_safe": int(accepted["false_safe"].sum()),
            "safe_agreement": int(accepted["safe_agreement"].sum()),
            "false_safe_rate_among_accepted": float(accepted["false_safe"].mean())
            if len(accepted)
            else None,
        }

    return {
        "overall": one(frame),
        "per_pde": {str(key): one(value) for key, value in frame.groupby("pde")},
        "per_pair": {str(key): one(value) for key, value in frame.groupby("pair")},
        "regime_changes_from_saved_replay": int(frame["saved_to_fresh_regime_changed"].sum())
        if "saved_to_fresh_regime_changed" in frame
        else None,
    }


def matching_sensitivity(
    frame: pd.DataFrame, accepted: pd.DataFrame, calipers: tuple[float, ...]
) -> list[dict[str, Any]]:
    records = []
    for caliper in calipers:
        matches = optimal_matches(accepted, caliper)
        if matches.empty:
            records.append({"caliper": caliper, "n_matches": 0})
            continue
        effect = matched_metric_summary(frame, matches, "log_tail_concentration_q99")
        records.append(
            {
                "caliper": float(caliper),
                "n_matches": int(len(matches)),
                "false_safe_retained_fraction": float(
                    len(matches) / max(1, int(accepted["false_safe"].sum()))
                ),
                "mean_paired_difference": effect["mean_paired_difference"],
                "probability_difference_positive": effect["probability_difference_positive"],
                "balance": {
                    covariate: standardized_mean_difference(
                        frame.loc[matches["fs_row_index"], covariate],
                        frame.loc[matches["control_row_index"], covariate],
                    )
                    for covariate in MATCH_COVARIATES
                },
            }
        )
    return records


def write_report(path: Path, payload: dict[str, Any]) -> None:
    counts = payload["counts"]["overall"]
    primary = payload["matched_effects"]["log_tail_concentration_q99"]
    boot = primary["grouped_bootstrap"]
    models = payload["grouped_oof"]["overall"]
    lines = [
        "# Critical-event fidelity analysis",
        "",
        "## Cohort",
        "",
        f"- Total model-condition evaluations: {counts['n']}",
        f"- Surrogate-accepted controls: {counts['surrogate_accepted']}",
        f"- False-safe controls: {counts['false_safe']}",
        f"- Safe-agreement controls: {counts['safe_agreement']}",
        "",
        "## Matched primary endpoint",
        "",
        f"- Matched pairs: {primary['n_matches']}",
        f"- Mean paired log tail-concentration difference: {primary['mean_paired_difference']:.6g}",
        f"- Grouped-bootstrap 95% CI: [{boot.get('ci95_low', float('nan')):.6g}, "
        f"{boot.get('ci95_high', float('nan')):.6g}]",
        f"- Probability of a positive paired difference: {primary['probability_difference_positive']:.3f}",
        "",
        "## Grouped out-of-fold models",
        "",
        "| Model | Log loss | Brier | AUROC | AP |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in models.items():
        lines.append(
            f"| {name} | {row['log_loss']:.4f} | {row['brier']:.4f} | "
            f"{row['auroc']:.4f} | {row['average_precision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Peak-time displacement is interpreted as a dynamic-misalignment diagnostic. "
            "For a horizon-wide maximum constraint, temporal displacement alone does not change the maximum-risk label.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.features_jsonl)
    frame = prepare_frame(rows)
    if int(args.require_full) and len(frame) != 2400:
        raise RuntimeError(f"Expected 2,400 critical-event rows, found {len(frame)}")
    accepted = frame[frame["surrogate_accepted"]].copy()
    matches = optimal_matches(accepted, float(args.match_caliper))
    if matches.empty:
        raise RuntimeError("No false-safe/safe-agreement matches were formed")

    balance = {}
    for covariate in MATCH_COVARIATES:
        before = standardized_mean_difference(
            accepted.loc[accepted["false_safe"], covariate],
            accepted.loc[accepted["safe_agreement"], covariate],
        )
        after = standardized_mean_difference(
            frame.loc[matches["fs_row_index"], covariate],
            frame.loc[matches["control_row_index"], covariate],
        )
        balance[covariate] = {"smd_before": before, "smd_after": after}

    effect_metrics = (
        "log_tail_concentration_q99",
        "critical_signed_underestimate_norm",
        "critical_abs_error_norm",
        "peak_time_displacement_frac",
        "peak_space_displacement_frac",
        "field_rel_l2_all",
        "risk_trace_mae_norm",
    )
    bootstrap_effects = cluster_bootstrap_matched_effects(
        accepted,
        effect_metrics,
        float(args.match_caliper),
        int(args.match_bootstrap),
        int(args.seed) + 1000,
    )
    effects = {}
    for metric in effect_metrics:
        summary = matched_metric_summary(frame, matches, metric)
        summary["grouped_bootstrap"] = bootstrap_effects[metric]
        effects[metric] = summary

    per_pde_effects = {}
    for pde, subset in accepted.groupby("pde"):
        local_matches = optimal_matches(subset, float(args.match_caliper))
        if local_matches.empty:
            continue
        local = matched_metric_summary(frame, local_matches, "log_tail_concentration_q99")
        local["grouped_bootstrap"] = cluster_bootstrap_matched_effects(
            subset,
            ("log_tail_concentration_q99",),
            float(args.match_caliper),
            int(args.match_bootstrap),
            int(args.seed) + 9000 + len(per_pde_effects),
        )["log_tail_concentration_q99"]
        per_pde_effects[str(pde)] = local

    predictions, oof = grouped_oof_predictions(accepted, int(args.n_splits), int(args.seed))
    repeated_oof_table, repeated_oof = repeated_grouped_oof_metrics(
        accepted,
        int(args.n_splits),
        int(args.cv_repeats),
        int(args.seed) + 30001,
    )
    deltas = [
        bootstrap_metric_deltas(
            predictions,
            ("aggregate", "extreme_unsigned"),
            int(args.metric_bootstrap),
            int(args.seed) + 20001,
        ),
        bootstrap_metric_deltas(
            predictions,
            ("aggregate", "extreme_signed"),
            int(args.metric_bootstrap),
            int(args.seed) + 20002,
        ),
    ]
    leave_pde = leave_one_pde_out(accepted)

    stable_accepted = accepted.copy()
    if "saved_to_fresh_regime_changed" in stable_accepted:
        stable_accepted = stable_accepted[
            stable_accepted["saved_to_fresh_regime_changed"].astype(int) == 0
        ]
    stable_matches = optimal_matches(stable_accepted, float(args.match_caliper))
    stability_sensitivity = {
        "n_accepted": int(len(stable_accepted)),
        "n_false_safe": int(stable_accepted["false_safe"].sum()),
        "n_matches": int(len(stable_matches)),
        "primary_effect": matched_metric_summary(
            frame, stable_matches, "log_tail_concentration_q99"
        ),
    }
    boundary_sensitivity = []
    for epsilon in (1e-4, 5e-4, 1e-3):
        subset = accepted[accepted["m_sur_norm"].astype(float) > epsilon]
        local_matches = optimal_matches(subset, float(args.match_caliper))
        boundary_sensitivity.append(
            {
                "minimum_normalized_surrogate_margin": epsilon,
                "n_accepted": int(len(subset)),
                "n_false_safe": int(subset["false_safe"].sum()),
                "n_matches": int(len(local_matches)),
                "primary_effect": matched_metric_summary(
                    frame, local_matches, "log_tail_concentration_q99"
                ),
            }
        )

    payload = {
        "analysis": "Critical-event fidelity under matched aggregate accuracy",
        "features_jsonl": str(args.features_jsonl.resolve()),
        "counts": count_summary(frame),
        "matching": {
            "method": "optimal one-to-one matching without replacement within pair and challenge tag",
            "covariates": list(MATCH_COVARIATES),
            "caliper_standard_deviations_per_coordinate": float(args.match_caliper),
            "n_matches": int(len(matches)),
            "n_false_safe_unmatched": int(accepted["false_safe"].sum() - len(matches)),
            "balance": balance,
        },
        "matched_effects": effects,
        "matched_primary_effect_per_pde": per_pde_effects,
        "grouped_oof": oof,
        "repeated_grouped_oof": repeated_oof,
        "grouped_oof_metric_deltas": deltas,
        "leave_one_pde_out": leave_pde,
        "matching_caliper_sensitivity": matching_sensitivity(
            frame, accepted, (0.25, 0.5, 1.0)
        ),
        "saved_replay_stability_sensitivity": stability_sensitivity,
        "surrogate_margin_boundary_sensitivity": boundary_sensitivity,
        "interpretation_constraint": (
            "Peak-time displacement is secondary dynamic evidence and is not treated as a direct "
            "cause of reversal under a horizon-wide maximum-risk label."
        ),
        "label_leakage_guard": (
            "label_defining_peak_gap_norm is excluded from every matching and predictive model."
        ),
    }
    (args.output_dir / "critical_event_summary.json").write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    frame.to_csv(args.output_dir / "critical_event_feature_table.csv", index=False)
    matches.to_csv(args.output_dir / "critical_event_matched_pairs.csv", index=False)
    predictions.to_csv(args.output_dir / "critical_event_grouped_predictions.csv", index=False)
    repeated_oof_table.to_csv(
        args.output_dir / "critical_event_repeated_grouped_metrics.csv", index=False
    )
    pd.DataFrame(leave_pde).to_csv(args.output_dir / "critical_event_leave_one_pde_out.csv", index=False)
    report_path = args.output_dir / "critical_event_results.md"
    write_report(report_path, payload)
    print(report_path.read_text(encoding="utf-8"), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-jsonl",
        type=Path,
        default=BASE
        / "external_data"
        / "analysis"
        / "critical_events"
        / "critical_event_features.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "outputs" / "critical_event_analysis" / "analysis",
    )
    parser.add_argument("--match-caliper", type=float, default=0.5)
    parser.add_argument("--match-bootstrap", type=int, default=1000)
    parser.add_argument("--metric-bootstrap", type=int, default=2000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--require-full", type=int, default=1)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

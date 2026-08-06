"""Audit the canonical discovery labels, optimizer feasibility metadata and critical-event replay.

The discovery labels in the manuscript come from ``external_data/decision_archives/discovery``.
critical-event recomputed trajectories in a later software process, so this audit keeps the
original labels fixed and treats later replays as numerical reproducibility
checks rather than silently redefining the discovery cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


BASE = Path(__file__).resolve().parents[1]
PDES = ("burgers", "grayscott", "kolmogorov")
MODELS = ("deeponet", "fno", "pino", "pinn")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regime(m_sur: float, m_hf: float) -> str:
    if m_sur >= 0.0 and m_hf >= 0.0:
        return "safe_agreement"
    if m_sur >= 0.0 and m_hf < 0.0:
        return "false_safe"
    if m_sur < 0.0 and m_hf >= 0.0:
        return "conservative_rejection"
    return "unsafe_agreement"


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": float(np.max(values)),
    }


def canonical_rows(cache_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_pair: dict[str, Any] = {}
    key_sets: dict[str, list[str]] = {}
    for pde in PDES:
        for model in MODELS:
            pair = f"{pde}:{model}"
            path = cache_root / pde / model / "challenge_conditions" / "decision_data.npz"
            with np.load(path, allow_pickle=True) as data:
                margins_sur = np.asarray(data["margin_sur"], dtype=np.float64)
                margins_hf = np.asarray(data["margin_hf"], dtype=np.float64)
                limits = np.asarray(data["z_limit"], dtype=np.float64)
                z_sur = np.asarray(data["z_sur"], dtype=np.float64)
                z_hf = np.asarray(data["z_hf"], dtype=np.float64)
                tags = np.asarray(data["tags"]).astype(str)
                key_sets[pair] = sorted(data.files)
                counts: Counter[str] = Counter()
                for index in range(len(margins_sur)):
                    label = regime(float(margins_sur[index]), float(margins_hf[index]))
                    counts[label] += 1
                    rows.append(
                        {
                            "pde": pde,
                            "model": model,
                            "pair": pair,
                            "ic_idx": index,
                            "tag": str(tags[index]),
                            "z_limit": float(limits[index]),
                            "z_sur": float(z_sur[index]),
                            "z_hf": float(z_hf[index]),
                            "m_sur": float(margins_sur[index]),
                            "m_hf": float(margins_hf[index]),
                            "regime": label,
                        }
                    )
                negative = margins_sur[margins_sur < 0.0]
                per_pair[pair] = {
                    "n": int(len(margins_sur)),
                    "counts": dict(counts),
                    "negative_surrogate_margin": int(len(negative)),
                    "negative_margin_quantiles": quantiles(negative) if len(negative) else None,
                    "cache_sha256": sha256(path),
                }

    pooled = Counter(row["regime"] for row in rows)
    by_pde: dict[str, Any] = {}
    for pde in PDES:
        local = [row for row in rows if row["pde"] == pde]
        negatives = np.asarray([row["m_sur"] for row in local if row["m_sur"] < 0.0])
        by_pde[pde] = {
            "n": len(local),
            "counts": dict(Counter(row["regime"] for row in local)),
            "negative_surrogate_margin": int(len(negatives)),
            "negative_margin_quantiles": quantiles(negatives),
        }

    solver_fields = (
        "solver_status",
        "return_status",
        "success",
        "iterations",
        "constraint_violation",
        "proxy_margin",
    )
    metadata = {
        "n": len(rows),
        "pooled_counts": dict(pooled),
        "negative_surrogate_margin": int(sum(row["m_sur"] < 0.0 for row in rows)),
        "by_pde": by_pde,
        "by_pair": per_pair,
        "solver_metadata_present": {
            field: any(field in keys for keys in key_sets.values()) for field in solver_fields
        },
        "implementation_audit": {
            "burgers": (
                "The stored control was selected under a one-shot local linearization and a "
                "smooth p-norm gradient constraint; the reported z_sur was recomputed from "
                "the nonlinear autoregressive surrogate rollout."
            ),
            "grayscott": (
                "Sequential linearized updates were checked by nonlinear surrogate rollout "
                "with a nonzero acceptance tolerance."
            ),
            "kolmogorov": (
                "Sequential linearized subproblems selected the stored control; the reported "
                "z_sur came from the final nonlinear surrogate rollout."
            ),
            "status_reconstructable": False,
            "status_note": (
                "The original NPZ artifacts do not contain optimizer return status or proxy "
                "constraint residuals. These fields cannot be reconstructed exactly without "
                "rerunning optimization and must not be inferred from final nonlinear margins."
            ),
        },
    }
    return rows, metadata


def audit_critical_event(
    canonical: list[dict[str, Any]], critical_event_path: Path, output_dir: Path
) -> dict[str, Any]:
    canonical_by_key = {
        (row["pde"], row["model"], int(row["ic_idx"])): row for row in canonical
    }
    critical_event_rows = read_jsonl(critical_event_path)
    relabelled: list[dict[str, Any]] = []
    source_changes: list[dict[str, Any]] = []
    cache_changes: list[dict[str, Any]] = []
    errors: list[float] = []
    for row in critical_event_rows:
        key = (str(row["pde"]), str(row["model"]), int(row["ic_idx"]))
        reference = canonical_by_key[key]
        updated = dict(row)
        updated["fresh_regime"] = str(row["regime"])
        updated["fresh_m_sur"] = float(row["m_sur"])
        updated["fresh_m_hf"] = float(row["m_hf"])
        updated["regime"] = reference["regime"]
        updated["m_sur"] = reference["m_sur"]
        updated["m_hf"] = reference["m_hf"]
        updated["z_sur_label_source"] = reference["z_sur"]
        updated["z_hf_label_source"] = reference["z_hf"]
        updated["label_source"] = "external_data/decision_archives/discovery"
        updated["fresh_to_canonical_regime_changed"] = int(
            updated["fresh_regime"] != updated["regime"]
        )
        error = float(row.get("z_sur_abs_error_to_saved_replay", np.nan))
        errors.append(error)
        updated["surrogate_replay_stable_1e_2"] = int(np.isfinite(error) and error <= 1e-2)
        relabelled.append(updated)
        if int(row.get("saved_to_fresh_regime_changed", 0)):
            source_changes.append(updated)
        if int(updated["fresh_to_canonical_regime_changed"]):
            cache_changes.append(updated)

    output_path = output_dir / "critical_event_extreme_event_features_canonical_labels.jsonl"
    write_jsonl(output_path, relabelled)
    canonical_counts = Counter(row["regime"] for row in relabelled)
    fresh_counts = Counter(row["fresh_regime"] for row in relabelled)
    by_pair: dict[str, Any] = {}
    for pair in sorted({str(row["pair"]) for row in relabelled}):
        local = [row for row in relabelled if row["pair"] == pair]
        by_pair[pair] = {
            "n": len(local),
            "canonical_false_safe": int(sum(row["regime"] == "false_safe" for row in local)),
            "fresh_false_safe": int(sum(row["fresh_regime"] == "false_safe" for row in local)),
            "changed": int(sum(row["fresh_to_canonical_regime_changed"] for row in local)),
            "stable_at_1e_2": int(sum(row["surrogate_replay_stable_1e_2"] for row in local)),
            "max_surrogate_replay_error": float(
                max(float(row["z_sur_abs_error_to_saved_replay"]) for row in local)
            ),
        }
    return {
        "source": str(critical_event_path.resolve()),
        "source_sha256": sha256(critical_event_path),
        "canonical_label_jsonl": str(output_path.resolve()),
        "canonical_counts": dict(canonical_counts),
        "fresh_counts": dict(fresh_counts),
        "fresh_to_canonical_regime_changes": len(cache_changes),
        "saved_to_fresh_regime_changes": len(source_changes),
        "surrogate_replay_error": quantiles(np.asarray(errors)),
        "unstable_above_1e_2": int(sum(not row["surrogate_replay_stable_1e_2"] for row in relabelled)),
        "changes": [
            {
                "pde": row["pde"],
                "model": row["model"],
                "ic_idx": row["ic_idx"],
                "canonical": row["regime"],
                "fresh": row["fresh_regime"],
                "z_sur_replay_error": row["z_sur_abs_error_to_saved_replay"],
            }
            for row in cache_changes
        ],
        "by_pair": by_pair,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    discovery = payload["discovery"]
    critical_event = payload["critical_event"]
    lines = [
        "# Integrity audit data-integrity audit",
        "",
        "## Canonical discovery cohort",
        "",
        f"- Canonical source: `{payload['canonical_cache_root']}`",
        f"- Model-condition evaluations: {discovery['n']}",
        f"- Regime counts: {discovery['pooled_counts']}",
        f"- Negative final nonlinear surrogate margins: {discovery['negative_surrogate_margin']}",
        "",
        "| PDE | Negative margin count | Minimum | Median | Maximum |",
        "|---|---:|---:|---:|---:|",
    ]
    for pde, row in discovery["by_pde"].items():
        q = row["negative_margin_quantiles"]
        lines.append(
            f"| {pde} | {row['negative_surrogate_margin']} | {q['minimum']:.6g} | "
            f"{q['median']:.6g} | {q['maximum']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Optimizer metadata",
            "",
            f"- Persisted solver fields: {discovery['solver_metadata_present']}",
            f"- Audit conclusion: {discovery['implementation_audit']['status_note']}",
            "",
            "## critical-event provenance",
            "",
            f"- Canonical-label counts: {critical_event['canonical_counts']}",
            f"- Fresh-replay counts: {critical_event['fresh_counts']}",
            f"- Fresh-to-canonical regime changes: {critical_event['fresh_to_canonical_regime_changes']}",
            f"- Saved-to-fresh regime changes: {critical_event['saved_to_fresh_regime_changes']}",
            f"- Fresh surrogate replay errors above 0.01: {critical_event['unstable_above_1e_2']}",
            "",
            "The manuscript discovery counts must continue to use the canonical cache. critical-event may use "
            "fresh trajectories for feature computation, but its labels are anchored to the canonical "
            "cache and replay-sensitive rows require a stated sensitivity analysis.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root", type=Path, default=BASE / "external_data/decision_archives/discovery"
    )
    parser.add_argument(
        "--critical_event-jsonl",
        type=Path,
        default=BASE
        / "external_data/analysis/critical_events"
        / "critical_event_features.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "outputs" / "audits" / "data_integrity",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    canonical, discovery = canonical_rows(args.cache_root)
    critical_event = audit_critical_event(canonical, args.critical_event_jsonl, args.output_dir)
    payload = {
        "audit": "Data-integrity audit",
        "canonical_cache_root": str(args.cache_root.resolve()),
        "discovery": discovery,
        "critical_event": critical_event,
    }
    json_path = args.output_dir / "integrity_audit.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "INTEGRITY_AUDIT.md", payload)
    print(json.dumps({"json": str(json_path), "summary": payload}, indent=2))


if __name__ == "__main__":
    main()

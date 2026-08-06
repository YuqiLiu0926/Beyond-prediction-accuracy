#!/usr/bin/env python3
"""Calibrate the final thermal numerical hierarchy from grouped query records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]


def read_queries(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"No numerical calibration queries were found in {path}")
    required = {"episode_id", "z_N0", "z_N1", "z_target"}
    missing = required - rows[0].keys()
    if missing:
        raise KeyError(f"Calibration queries are missing fields: {sorted(missing)}")
    return rows


def grouped_maxima(values: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(max(group)) for key, group in sorted(values.items())}


def calibrate(
    rows: list[dict[str, Any]], refinement_ratio: float, nominal_order: float
) -> dict[str, Any]:
    gamma = 1.0 / (refinement_ratio**nominal_order - 1.0)

    n1_residuals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        z_n0 = float(row["z_N0"])
        z_n1 = float(row["z_N1"])
        z_target = float(row["z_target"])
        defect = abs(z_n1 - z_n0)
        row["resolution_defect"] = defect
        n1_residuals[str(row["episode_id"])].append(
            z_target - z_n1 - gamma * defect
        )
    n1_episode_maxima = grouped_maxima(n1_residuals)
    n1_allowance = max(0.0, max(n1_episode_maxima.values()))

    n0_residuals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        z_n0 = float(row["z_N0"])
        z_n1 = float(row["z_N1"])
        defect = float(row["resolution_defect"])
        n1_upper = z_n1 + gamma * defect + n1_allowance
        n0_residuals[str(row["episode_id"])].append(n1_upper - z_n0)
    n0_episode_maxima = grouped_maxima(n0_residuals)
    n0_allowance = max(0.0, max(n0_episode_maxima.values()))

    episode_count = len(n0_episode_maxima)
    if episode_count != len(n1_episode_maxima):
        raise RuntimeError("N0 and N1 calibration groups do not match")
    return {
        "schema_version": 2,
        "status": "CALIBRATED_FROM_COMMISSIONING_EPISODES",
        "calibration_unit": "episode maximum over evaluated sequence queries",
        "query_count": len(rows),
        "episode_count": episode_count,
        "empirical_rank_coverage": episode_count / (episode_count + 1.0),
        "formal_95_percent_claim": False,
        "numerical_levels": {
            "N0": {"spatial_grid_size": 32, "substeps_per_control_interval": 10},
            "N1": {"spatial_grid_size": 64, "substeps_per_control_interval": 10},
            "target": {"spatial_grid_size": 64, "substeps_per_control_interval": 20},
        },
        "refinement_ratio": refinement_ratio,
        "nominal_order": nominal_order,
        "richardson_gamma": gamma,
        "N0_upper_risk_formula": "z_N0 + N0_one_sided_allowance",
        "N1_upper_risk_formula": (
            "z_N1 + richardson_gamma * abs(z_N1 - z_N0) "
            "+ N1_one_sided_allowance"
        ),
        "N0_one_sided_allowance": n0_allowance,
        "N1_one_sided_allowance": n1_allowance,
        "N0_episode_maxima": n0_episode_maxima,
        "N1_episode_maxima": n1_episode_maxima,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=BASE
        / "external_data/numerical_qualification/thermal/allowance_queries.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE
        / "outputs/numerical_qualification/thermal/one_sided_allowance.json",
    )
    parser.add_argument("--refinement-ratio", type=float, default=2.0)
    parser.add_argument("--nominal-order", type=float, default=2.0)
    args = parser.parse_args()

    payload = calibrate(
        read_queries(args.queries), args.refinement_ratio, args.nominal_order
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

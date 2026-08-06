#!/usr/bin/env python3
"""recoverability reference study: state-conditioned physical recoverability reference audit.

The coarse-PDE search represents an online-capable generator. The HF search is an
offline diagnostic upper bound only and is never presented as a deployable policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from continuation_library import candidate_library, make_evaluator, physical_risk


BASE = Path(__file__).resolve().parents[1]
PDES = ("burgers", "grayscott", "kolmogorov")


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or any(value < 1 for value in values):
        raise ValueError(f"Expected positive integers, got {text!r}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_blocks(alpha: np.ndarray, horizon: int) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    edges = np.linspace(0, horizon, len(alpha) + 1, dtype=np.int64)
    control = np.empty(horizon, dtype=np.float64)
    for index, value in enumerate(alpha):
        control[edges[index] : edges[index + 1]] = value
    return control


def reduce_blocks(control: np.ndarray, n_blocks: int) -> np.ndarray:
    control = np.asarray(control, dtype=np.float64).reshape(-1)
    edges = np.linspace(0, len(control), n_blocks + 1, dtype=np.int64)
    return np.asarray([np.mean(control[edges[i] : edges[i + 1]]) for i in range(n_blocks)])


@dataclass
class CEMResult:
    best_alpha: np.ndarray
    best_control: np.ndarray
    best_value: float
    finalists: list[tuple[float, np.ndarray]]
    evaluations: int
    trace: list[dict[str, float | int]]


def cem_minimize(
    objective: Callable[[np.ndarray], float],
    horizon: int,
    n_blocks: int,
    lower: float,
    upper: float,
    initial_means: list[np.ndarray],
    seed: int,
    population: int,
    iterations: int,
    elite_fraction: float,
    finalists: int,
) -> CEMResult:
    if population < 4 or iterations < 1:
        raise ValueError("CEM budget is too small")
    elite_n = max(2, int(math.ceil(population * elite_fraction)))
    span = float(upper - lower)
    rng = np.random.default_rng(seed)
    archive: list[tuple[float, np.ndarray]] = []
    trace: list[dict[str, float | int]] = []
    evaluations = 0

    for restart, initial in enumerate(initial_means):
        mean = np.clip(np.asarray(initial, dtype=np.float64), lower, upper)
        if mean.shape != (n_blocks,):
            raise ValueError(f"Bad initial mean shape {mean.shape}, expected {(n_blocks,)}")
        std = np.full(n_blocks, 0.45 * span, dtype=np.float64)
        for iteration in range(iterations):
            samples = mean[None, :] + rng.standard_normal((population, n_blocks)) * std[None, :]
            samples = np.clip(samples, lower, upper)
            samples[0] = mean
            if population >= 4:
                samples[1] = 0.0
                samples[2] = lower
                samples[3] = upper
            values = np.empty(population, dtype=np.float64)
            for index, alpha in enumerate(samples):
                value = float(objective(expand_blocks(alpha, horizon)))
                if not math.isfinite(value):
                    raise RuntimeError("Non-finite CEM objective")
                values[index] = value
                archive.append((value, alpha.copy()))
            evaluations += population
            order = np.argsort(values)
            elite = samples[order[:elite_n]]
            elite_mean = np.mean(elite, axis=0)
            elite_std = np.std(elite, axis=0)
            mean = 0.25 * mean + 0.75 * elite_mean
            std = np.maximum(0.25 * std + 0.75 * elite_std, 0.015 * span)
            trace.append(
                {
                    "restart": restart,
                    "iteration": iteration,
                    "best": float(values[order[0]]),
                    "elite_mean": float(np.mean(values[order[:elite_n]])),
                    "std_mean": float(np.mean(std)),
                }
            )

    archive.sort(key=lambda item: item[0])
    unique: list[tuple[float, np.ndarray]] = []
    seen: set[bytes] = set()
    for value, alpha in archive:
        key = np.round(alpha, decimals=8).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append((float(value), expand_blocks(alpha, horizon)))
        if len(unique) >= finalists:
            break
    best_value, best_alpha = archive[0]
    return CEMResult(
        best_alpha=best_alpha,
        best_control=expand_blocks(best_alpha, horizon),
        best_value=float(best_value),
        finalists=unique,
        evaluations=evaluations,
        trace=trace,
    )


def self_check() -> None:
    expanded = expand_blocks(np.asarray([-1.0, 0.5, 1.0]), 10)
    assert len(expanded) == 10 and expanded[0] == -1.0 and expanded[-1] == 1.0
    reduced = reduce_blocks(expanded, 3)
    assert np.max(np.abs(reduced - np.asarray([-1.0, 0.5, 1.0]))) < 1e-12

    target = np.asarray([0.25, -0.35, 0.1])
    result = cem_minimize(
        objective=lambda control: float(np.sum((reduce_blocks(control, 3) - target) ** 2)),
        horizon=12,
        n_blocks=3,
        lower=-1.0,
        upper=1.0,
        initial_means=[np.zeros(3), np.full(3, 0.5)],
        seed=7,
        population=48,
        iterations=12,
        elite_fraction=0.15,
        finalists=5,
    )
    assert result.best_value < 2e-3, result.best_value
    print(json.dumps({"self_check": "PASS", "toy_best": result.best_value}))


def summarize(rows: list[dict[str, Any]], boundary_scale: float, repair_gate: int) -> dict[str, Any]:
    unsafe = [row for row in rows if not int(row["continuation_only_hf_safe"])]
    repaired_oracle = [row for row in unsafe if int(row["hf_oracle_safe"])]
    repaired_online = [row for row in unsafe if int(row["lf_selected_hf_safe"])]
    by_pde = {}
    for pde in PDES:
        local = [row for row in rows if row["pde"] == pde]
        local_unsafe = [row for row in local if not int(row["continuation_only_hf_safe"])]
        if not local:
            continue
        by_pde[pde] = {
            "targets": len(local),
            "continuation_only_hf_unsafe": len(local_unsafe),
            "k15_hf_oracle_repaired": sum(int(row["k15_hf_oracle_safe"]) for row in local_unsafe),
            "lf_selected_repaired": sum(int(row["lf_selected_hf_safe"]) for row in local_unsafe),
            "hf_oracle_repaired": sum(int(row["hf_oracle_safe"]) for row in local_unsafe),
        }
    return {
        "boundary": f"Z <= {boundary_scale} * Z(initial_state)",
        "target_rows": len(rows),
        "continuation_only_hf_unsafe": len(unsafe),
        "k15_hf_oracle_repaired": sum(int(row["k15_hf_oracle_safe"]) for row in unsafe),
        "lf_selected_repaired": len(repaired_online),
        "hf_oracle_repaired": len(repaired_oracle),
        "oracle_repair_rate": len(repaired_oracle) / max(1, len(unsafe)),
        "proceed_gate": {
            "required_repaired": repair_gate,
            "passed": len(repaired_oracle) >= repair_gate,
        },
        "by_pde": by_pde,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE / "external_data/decision_archives/evaluation")
    parser.add_argument("--continuation-rows", type=Path, default=BASE / "external_data/analysis/recoverability/continuation_rows.jsonl")
    parser.add_argument("--split", type=Path, default=BASE / "external_data/analysis/recoverability/state_split.json")
    parser.add_argument("--outdir", type=Path, default=BASE / "outputs/recoverability_reference")
    parser.add_argument("--target", choices=("uncertified", "unsafe"), default="uncertified")
    parser.add_argument("--pdes", default="burgers,grayscott")
    parser.add_argument("--blocks", default="2,4,8")
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--elite-fraction", type=float, default=0.15)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--boundary-scale", type=float, default=1.20)
    parser.add_argument("--coarse-substeps", type=int, default=2)
    parser.add_argument("--repair-gate", type=int, default=8)
    parser.add_argument("--ic-seed", type=int, default=20260731)
    parser.add_argument("--ic-mode", default="ood_amplified")
    parser.add_argument("--n-ics", type=int, default=100)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--ckpt-root", type=Path, default=BASE / "external_data/checkpoints/discovery")
    parser.add_argument("--model-data-root", type=Path, default=BASE / "external_data/datasets/discovery")
    parser.add_argument("--train-script", type=Path, default=BASE / "scripts/train_discovery_surrogates.py")
    parser.add_argument("--burgers-script", type=Path, default=BASE / "scripts/generate_burgers_decisions.py")
    parser.add_argument("--grayscott-script", type=Path, default=BASE / "scripts/generate_gray_scott_decisions.py")
    parser.add_argument("--kolmogorov-script", type=Path, default=BASE / "scripts/generate_kolmogorov_decisions.py")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.restarts != 2:
        raise ValueError("The frozen diagnostic uses exactly two starts: K15-best and zero")

    pdes = tuple(value.strip() for value in args.pdes.split(",") if value.strip())
    if any(pde not in PDES for pde in pdes):
        raise ValueError(args.pdes)
    blocks = parse_ints(args.blocks)
    split = json.loads(args.split.read_text(encoding="utf-8"))["assignments"]
    continuation_all = list(iter_jsonl(args.continuation_rows))
    continuation = {(str(row["pde"]), int(row["ic_idx"])): row for row in continuation_all}
    targets = []
    for pde in pdes:
        for ic_idx in split[pde]["test_ic_idx"]:
            row = continuation[(pde, int(ic_idx))]
            selected = not int(row["continuation_certified"]) if args.target == "uncertified" else not int(row["continuation_hf_safe"])
            if selected:
                targets.append((pde, int(ic_idx)))
    if args.target == "uncertified" and len(targets) != 12:
        raise RuntimeError(f"Expected 12 frozen uncertified targets, found {len(targets)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows_path = args.outdir / "reference_search_rows.jsonl"
    if not int(args.resume) and rows_path.exists():
        rows_path.unlink()
    existing = list(iter_jsonl(rows_path))
    done = {(str(row["pde"]), int(row["ic_idx"])) for row in existing}
    started = time.perf_counter()

    for pde in pdes:
        local_targets = [ic_idx for local_pde, ic_idx in targets if local_pde == pde]
        if not local_targets:
            continue
        data_path = args.data_root / pde / "deeponet" / args.ic_mode / "decision_data.npz"
        with np.load(data_path, allow_pickle=True) as data:
            states = np.asarray(data["initial_states"], dtype=np.float64)
            z0s = np.asarray(data["z0"], dtype=np.float64)
            horizon = int(np.asarray(data["U_all"]).shape[1])
        evaluator = make_evaluator(args, pde, horizon)
        names, library = candidate_library(horizon, evaluator.umin, evaluator.umax)

        for ic_idx in local_targets:
            if (pde, ic_idx) in done:
                continue
            row_started = time.perf_counter()
            x0 = states[ic_idx]
            z0 = float(z0s[ic_idx])
            boundary = float(args.boundary_scale) * z0
            regenerated = np.asarray(evaluator.x0s[ic_idx], dtype=np.float64)
            state_error = float(np.max(np.abs(regenerated - x0)))
            if state_error > 1e-6:
                raise RuntimeError(f"Initial-state mismatch {pde}:{ic_idx}: {state_error}")

            z_lf_k15 = np.asarray(
                [physical_risk(evaluator, x0, control, "coarse", args.coarse_substeps) for control in library]
            )
            z_hf_k15 = np.asarray(
                [physical_risk(evaluator, x0, control, "hf", args.coarse_substeps) for control in library]
            )
            continuation_only_choice = int(np.argmin(z_lf_k15[:5]))
            frozen = continuation[(pde, ic_idx)]
            frozen_lf = np.asarray(frozen["z_lf_candidates"], dtype=np.float64)
            continuation_only_low_fidelity_error = float(np.max(np.abs(z_lf_k15[:5] - frozen_lf)))
            continuation_only_hf_error = abs(float(z_hf_k15[continuation_only_choice]) - float(frozen["z_hf_selected"]))
            if continuation_only_low_fidelity_error > 1e-8 or continuation_only_hf_error > 1e-8 or continuation_only_choice != int(frozen["selected_index"]):
                raise RuntimeError(
                    f"K5 replay mismatch {pde}:{ic_idx}: lf={continuation_only_low_fidelity_error} hf={continuation_only_hf_error} choice={continuation_only_choice}"
                )

            searches: list[dict[str, Any]] = []
            lf_controls: list[tuple[float, np.ndarray, int]] = []
            hf_controls: list[tuple[float, np.ndarray, int]] = []
            for n_blocks in blocks:
                k15_seed = reduce_blocks(library[int(np.argmin(z_lf_k15))], n_blocks)
                initial_means = [k15_seed, np.zeros(n_blocks, dtype=np.float64)]

                lf_result = cem_minimize(
                    objective=lambda control: physical_risk(
                        evaluator, x0, control, "coarse", args.coarse_substeps
                    ),
                    horizon=horizon,
                    n_blocks=n_blocks,
                    lower=evaluator.umin,
                    upper=evaluator.umax,
                    initial_means=initial_means,
                    seed=args.seed + 10000 * ic_idx + 101 * n_blocks,
                    population=args.population,
                    iterations=args.iterations,
                    elite_fraction=args.elite_fraction,
                    finalists=args.finalists,
                )
                lf_best_hf = physical_risk(
                    evaluator, x0, lf_result.best_control, "hf", args.coarse_substeps
                )
                finalist_hf = [
                    physical_risk(evaluator, x0, control, "hf", args.coarse_substeps)
                    for _, control in lf_result.finalists
                ]
                for (lf_value, control), hf_value in zip(lf_result.finalists, finalist_hf):
                    lf_controls.append((float(lf_value), control, n_blocks))
                    hf_controls.append((float(hf_value), control, n_blocks))

                hf_result = cem_minimize(
                    objective=lambda control: physical_risk(
                        evaluator, x0, control, "hf", args.coarse_substeps
                    ),
                    horizon=horizon,
                    n_blocks=n_blocks,
                    lower=evaluator.umin,
                    upper=evaluator.umax,
                    initial_means=initial_means,
                    seed=args.seed + 10000 * ic_idx + 101 * n_blocks + 53,
                    population=args.population,
                    iterations=args.iterations,
                    elite_fraction=args.elite_fraction,
                    finalists=args.finalists,
                )
                hf_best_lf = physical_risk(
                    evaluator, x0, hf_result.best_control, "coarse", args.coarse_substeps
                )
                searches.append(
                    {
                        "n_blocks": n_blocks,
                        "lf_cem": {
                            "z_lf": lf_result.best_value,
                            "z_hf": float(lf_best_hf),
                            "safe_hf": int(lf_best_hf <= boundary),
                            "evaluations": lf_result.evaluations,
                            "alpha": lf_result.best_alpha.tolist(),
                            "trace": lf_result.trace,
                        },
                        "hf_oracle_cem": {
                            "z_hf": hf_result.best_value,
                            "z_lf": float(hf_best_lf),
                            "safe_hf": int(hf_result.best_value <= boundary),
                            "evaluations": hf_result.evaluations,
                            "alpha": hf_result.best_alpha.tolist(),
                            "trace": hf_result.trace,
                        },
                    }
                )

            lf_selected = min(lf_controls + [(float(np.min(z_lf_k15)), library[int(np.argmin(z_lf_k15))], 0)], key=lambda item: item[0])
            lf_selected_hf = physical_risk(
                evaluator, x0, lf_selected[1], "hf", args.coarse_substeps
            )
            hf_oracle_values = [float(np.min(z_hf_k15))]
            hf_oracle_values.extend(value for value, _, _ in hf_controls)
            hf_oracle_values.extend(float(search["hf_oracle_cem"]["z_hf"]) for search in searches)
            hf_oracle = float(min(hf_oracle_values))
            result_row = {
                "pde": pde,
                "ic_idx": ic_idx,
                "z0": z0,
                "boundary": boundary,
                "state_error": state_error,
                "continuation_only_selected_name": names[continuation_only_choice],
                "continuation_only_z_lf": float(z_lf_k15[continuation_only_choice]),
                "continuation_only_z_hf": float(z_hf_k15[continuation_only_choice]),
                "continuation_only_hf_safe": int(z_hf_k15[continuation_only_choice] <= boundary),
                "continuation_only_frozen_certified": int(frozen["continuation_certified"]),
                "continuation_only_replay_lf_max_error": continuation_only_low_fidelity_error,
                "continuation_only_replay_hf_error": continuation_only_hf_error,
                "k15_lf_selected_z_hf": float(z_hf_k15[int(np.argmin(z_lf_k15))]),
                "k15_hf_oracle_z_hf": float(np.min(z_hf_k15)),
                "k15_hf_oracle_safe": int(np.min(z_hf_k15) <= boundary),
                "lf_selected_n_blocks": int(lf_selected[2]),
                "lf_selected_z_lf": float(lf_selected[0]),
                "lf_selected_z_hf": float(lf_selected_hf),
                "lf_selected_hf_safe": int(lf_selected_hf <= boundary),
                "hf_oracle_z_hf": hf_oracle,
                "hf_oracle_safe": int(hf_oracle <= boundary),
                "searches": searches,
                "elapsed_seconds": time.perf_counter() - row_started,
            }
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result_row, sort_keys=True) + "\n")
            print(
                f"[recoverability reference study] {pde}:{ic_idx} K5={result_row['continuation_only_z_hf'] / z0:.4f} "
                f"LF={result_row['lf_selected_z_hf'] / z0:.4f} "
                f"oracle={result_row['hf_oracle_z_hf'] / z0:.4f} "
                f"safe={result_row['hf_oracle_safe']}",
                flush=True,
            )

    rows = list(iter_jsonl(rows_path))
    target_set = set(targets)
    selected_rows = [row for row in rows if (str(row["pde"]), int(row["ic_idx"])) in target_set]
    keys = {(str(row["pde"]), int(row["ic_idx"])) for row in selected_rows}
    if keys != target_set or len(selected_rows) != len(target_set):
        raise RuntimeError(f"Incomplete recoverability reference study rows={len(selected_rows)} targets={len(target_set)}")
    summary = {
        "protocol": "recoverability reference study state-conditioned physics recovery oracle",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "target": args.target,
        "target_keys": [f"{pde}:{ic_idx}" for pde, ic_idx in targets],
        "blocks": blocks,
        "cem": {
            "population": args.population,
            "iterations": args.iterations,
            "restarts": args.restarts,
            "elite_fraction": args.elite_fraction,
            "finalists": args.finalists,
            "seed": args.seed,
        },
        "coarse_substeps": args.coarse_substeps,
        "online_hf_used": False,
        "hf_oracle_role": "offline repairability upper bound only",
        "metrics": summarize(selected_rows, args.boundary_scale, args.repair_gate),
        "elapsed_seconds": time.perf_counter() - started,
        "rows_path": str(rows_path),
        "rows_sha256": sha256(rows_path),
    }
    summary_path = args.outdir / "reference_search_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary": str(summary_path), **summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()

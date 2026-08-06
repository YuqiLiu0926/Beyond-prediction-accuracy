"""Recoverability: map finite-horizon recoverability across available control families.

The experiment is an offline HF audit. It does not provide future HF
trajectories to the deployed controller. Two cohorts are kept separate:

1. the ten previously identified states whose K=5 continuation was HF unsafe;
2. a newly generated boundary-stress cohort selected with an independent seed.

The measured surface is conditional on the searched control families and
finite horizon. It is not presented as a proof that every admissible control
has been exhausted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from continuation_library import candidate_library, make_evaluator, physical_risk
from reference_recoverability_search import cem_minimize, reduce_blocks, self_check as cem_self_check


BASE = Path(__file__).resolve().parents[1]
PDES = ("burgers", "grayscott", "kolmogorov")


def parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError(text)
    return values


def parse_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError(text)
    return values


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def state_hash(state: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(state, dtype=np.float32).tobytes()).hexdigest()


def evaluator_args(args: argparse.Namespace, seed: int, n_ics: int) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=args.data_root,
        ckpt_root=args.ckpt_root,
        model_data_root=args.model_data_root,
        train_script=args.train_script,
        burgers_script=args.burgers_script,
        grayscott_script=args.grayscott_script,
        kolmogorov_script=args.kolmogorov_script,
        ic_mode=args.ic_mode,
        device=args.device,
        ic_seed=seed,
        n_ics=n_ics,
    )


def evaluate_library(
    evaluator: Any,
    state: np.ndarray,
    horizon: int,
    authority: float,
    dense_constants: int,
    coarse_substeps: int,
) -> dict[str, dict[str, Any]]:
    lower = float(evaluator.umin) * authority
    upper = float(evaluator.umax) * authority
    names, library = candidate_library(horizon, lower, upper)
    library_risk = np.asarray(
        [physical_risk(evaluator, state, control, "hf", coarse_substeps) for control in library],
        dtype=np.float64,
    )
    amplitudes = np.linspace(lower, upper, dense_constants, dtype=np.float64)
    constant_risk = np.asarray(
        [
            physical_risk(
                evaluator,
                state,
                np.full(horizon, amplitude, dtype=np.float64),
                "hf",
                coarse_substeps,
            )
            for amplitude in amplitudes
        ],
        dtype=np.float64,
    )
    continuation_only = int(np.argmin(library_risk[:5]))
    k15 = int(np.argmin(library_risk))
    dense = int(np.argmin(constant_risk))
    return {
        "K5": {
            "best_risk": float(library_risk[continuation_only]),
            "control_name": names[continuation_only],
            "control_amplitude": None,
            "candidate_count": 5,
        },
        "K15": {
            "best_risk": float(library_risk[k15]),
            "control_name": names[k15],
            "control_amplitude": None,
            "candidate_count": len(library),
        },
        "dense_constant": {
            "best_risk": float(constant_risk[dense]),
            "control_name": "constant",
            "control_amplitude": float(amplitudes[dense]),
            "candidate_count": dense_constants,
        },
    }


def select_validation_states(
    evaluator: Any,
    n_select: int,
    horizon: int,
    boundary_scale: float,
    coarse_substeps: int,
) -> list[dict[str, Any]]:
    records = []
    names, library = candidate_library(horizon, evaluator.umin, evaluator.umax)
    for ic_idx, (state, z0) in enumerate(zip(evaluator.x0s, evaluator.z0s)):
        risks = np.asarray(
            [physical_risk(evaluator, state, control, "hf", coarse_substeps) for control in library[:5]],
            dtype=np.float64,
        )
        choice = int(np.argmin(risks))
        normalized_margin = boundary_scale - float(risks[choice]) / float(z0)
        records.append(
            {
                "ic_idx": ic_idx,
                "z0": float(z0),
                "continuation_only_best_risk": float(risks[choice]),
                "continuation_only_name": names[choice],
                "normalized_margin": normalized_margin,
                "continuation_only_safe": int(normalized_margin >= 0.0),
                "state_hash": state_hash(state),
            }
        )
    safe = sorted((row for row in records if row["continuation_only_safe"]), key=lambda row: abs(row["normalized_margin"]))
    unsafe = sorted((row for row in records if not row["continuation_only_safe"]), key=lambda row: abs(row["normalized_margin"]))
    selected = safe[: n_select // 2] + unsafe[: n_select - n_select // 2]
    used = {row["ic_idx"] for row in selected}
    if len(selected) < n_select:
        remaining = sorted(
            (row for row in records if row["ic_idx"] not in used),
            key=lambda row: abs(row["normalized_margin"]),
        )
        selected.extend(remaining[: n_select - len(selected)])
    return sorted(selected, key=lambda row: row["ic_idx"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["cohort"], row["horizon"], row["authority"], row["family"])].append(row)
    surface = []
    for (cohort, horizon, authority, family), local in sorted(groups.items()):
        surface.append(
            {
                "cohort": cohort,
                "horizon": horizon,
                "authority": authority,
                "family": family,
                "n_states": len(local),
                "recoverable_states": int(sum(row["safe"] for row in local)),
                "recoverable_fraction": float(np.mean([row["safe"] for row in local])),
                "median_best_risk_ratio": float(np.median([row["risk_ratio_to_z0"] for row in local])),
            }
        )
    return {"surface": surface}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=BASE / "outputs/recoverability_surface",
    )
    parser.add_argument("--horizons", default="25,50,75,100")
    parser.add_argument("--authorities", default="1,1.25,1.5,2,3,4")
    parser.add_argument("--cem-authorities", default="1,2,4")
    parser.add_argument("--cem-blocks", default="2,4,8")
    parser.add_argument("--dense-constants", type=int, default=33)
    parser.add_argument("--cem-population", type=int, default=24)
    parser.add_argument("--cem-iterations", type=int, default=6)
    parser.add_argument("--cem-elite-fraction", type=float, default=0.20)
    parser.add_argument("--cem-finalists", type=int, default=4)
    parser.add_argument("--boundary-scale", type=float, default=1.20)
    parser.add_argument("--coarse-substeps", type=int, default=2)
    parser.add_argument("--primary-seed", type=int, default=20260731)
    parser.add_argument("--validation-seed", type=int, default=20260804)
    parser.add_argument("--validation-pool", type=int, default=60)
    parser.add_argument("--validation-per-pde", type=int, default=4)
    parser.add_argument("--ic-mode", default="ood_amplified")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--skip-cem", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--data-root", type=Path, default=BASE / "external_data" / "decision_archives" / "evaluation")
    parser.add_argument("--ckpt-root", type=Path, default=BASE / "external_data")
    parser.add_argument("--model-data-root", type=Path, default=BASE / "external_data")
    parser.add_argument("--train-script", type=Path, default=BASE / "scripts" / "train_discovery_surrogates.py")
    parser.add_argument("--burgers-script", type=Path, default=BASE / "scripts/generate_burgers_decisions.py")
    parser.add_argument("--grayscott-script", type=Path, default=BASE / "scripts/generate_gray_scott_decisions.py")
    parser.add_argument("--kolmogorov-script", type=Path, default=BASE / "scripts/generate_kolmogorov_decisions.py")
    args = parser.parse_args()
    if args.self_check:
        cem_self_check()
        return

    horizons = parse_ints(args.horizons)
    authorities = parse_floats(args.authorities)
    cem_authorities = parse_floats(args.cem_authorities)
    cem_blocks = parse_ints(args.cem_blocks)
    if max(horizons) != 100:
        raise ValueError("The current experiment expects the original 100-step horizon in the grid")

    args.outdir.mkdir(parents=True, exist_ok=True)
    surface_path = args.outdir / "surface_rows.jsonl"
    cem_path = args.outdir / "cem_rows.jsonl"
    selection_path = args.outdir / "validation_selection.json"
    if not args.resume:
        for path in (surface_path, cem_path, selection_path):
            if path.exists():
                path.unlink()

    existing_surface = {
        (row["cohort"], row["pde"], row["ic_idx"], row["horizon"], row["authority"], row["family"])
        for row in iter_jsonl(surface_path)
    }
    existing_cem = {
        (row["pde"], row["ic_idx"], row["authority"], row["n_blocks"])
        for row in iter_jsonl(cem_path)
    }
    primary_rows = list(
        iter_jsonl(BASE / "external_data/analysis/recoverability/reference_search_rows.jsonl")
    )
    primary_by_pde = {
        pde: sorted(int(row["ic_idx"]) for row in primary_rows if row["pde"] == pde and not int(row["continuation_only_hf_safe"]))
        for pde in PDES
    }
    started = time.perf_counter()
    validation_selection: dict[str, Any] = {}

    for pde in PDES:
        cohorts: list[tuple[str, Any, list[int]]] = []
        if primary_by_pde[pde]:
            evaluator = make_evaluator(
                evaluator_args(args, args.primary_seed, 100), pde, max(horizons)
            )
            cohorts.append(("mechanism", evaluator, primary_by_pde[pde]))

        validation_evaluator = make_evaluator(
            evaluator_args(args, args.validation_seed, args.validation_pool), pde, max(horizons)
        )
        selected = select_validation_states(
            validation_evaluator,
            args.validation_per_pde,
            max(horizons),
            args.boundary_scale,
            args.coarse_substeps,
        )
        validation_selection[pde] = selected
        cohorts.append(("independent_boundary_stress", validation_evaluator, [row["ic_idx"] for row in selected]))

        for cohort, local_evaluator, indices in cohorts:
            for ic_idx in indices:
                state = np.asarray(local_evaluator.x0s[ic_idx], dtype=np.float64)
                z0 = float(local_evaluator.z0s[ic_idx])
                boundary = args.boundary_scale * z0
                for horizon in horizons:
                    for authority in authorities:
                        expected_surface_keys = {
                            (cohort, pde, ic_idx, horizon, authority, family)
                            for family in ("K5", "K15", "dense_constant")
                        }
                        if expected_surface_keys.issubset(existing_surface):
                            continue
                        values = evaluate_library(
                            local_evaluator,
                            state,
                            horizon,
                            authority,
                            args.dense_constants,
                            args.coarse_substeps,
                        )
                        for family, result in values.items():
                            key = (cohort, pde, ic_idx, horizon, authority, family)
                            if key in existing_surface:
                                continue
                            row = {
                                "cohort": cohort,
                                "pde": pde,
                                "ic_idx": ic_idx,
                                "seed": args.primary_seed if cohort == "mechanism" else args.validation_seed,
                                "state_hash": state_hash(state),
                                "horizon": horizon,
                                "authority": authority,
                                "family": family,
                                "z0": z0,
                                "boundary": boundary,
                                **result,
                            }
                            row["risk_ratio_to_z0"] = row["best_risk"] / z0
                            row["safe"] = int(row["best_risk"] <= boundary)
                            append_jsonl(surface_path, row)
                            existing_surface.add(key)
                        print(
                            f"[Recoverability grid] {cohort} {pde}:{ic_idx} H={horizon} A={authority:g} "
                            f"best={min(v['best_risk'] for v in values.values()) / z0:.4f}",
                            flush=True,
                        )

                if cohort == "mechanism" and not args.skip_cem:
                    for authority in cem_authorities:
                        expected_cem_keys = {
                            (pde, ic_idx, authority, n_blocks) for n_blocks in cem_blocks
                        }
                        if expected_cem_keys.issubset(existing_cem):
                            continue
                        lower = float(local_evaluator.umin) * authority
                        upper = float(local_evaluator.umax) * authority
                        dense_amplitudes = np.linspace(lower, upper, args.dense_constants)
                        dense_risks = np.asarray(
                            [
                                physical_risk(
                                    local_evaluator,
                                    state,
                                    np.full(max(horizons), amplitude),
                                    "hf",
                                    args.coarse_substeps,
                                )
                                for amplitude in dense_amplitudes
                            ]
                        )
                        best_constant = np.full(
                            max(horizons), dense_amplitudes[int(np.argmin(dense_risks))]
                        )
                        for n_blocks in cem_blocks:
                            key = (pde, ic_idx, authority, n_blocks)
                            if key in existing_cem:
                                continue
                            result = cem_minimize(
                                objective=lambda control: physical_risk(
                                    local_evaluator,
                                    state,
                                    control,
                                    "hf",
                                    args.coarse_substeps,
                                ),
                                horizon=max(horizons),
                                n_blocks=n_blocks,
                                lower=lower,
                                upper=upper,
                                initial_means=[
                                    reduce_blocks(best_constant, n_blocks),
                                    np.zeros(n_blocks, dtype=np.float64),
                                ],
                                seed=args.validation_seed + 10000 * ic_idx + 101 * n_blocks + int(100 * authority),
                                population=args.cem_population,
                                iterations=args.cem_iterations,
                                elite_fraction=args.cem_elite_fraction,
                                finalists=args.cem_finalists,
                            )
                            row = {
                                "cohort": "mechanism",
                                "pde": pde,
                                "ic_idx": ic_idx,
                                "authority": authority,
                                "horizon": max(horizons),
                                "n_blocks": n_blocks,
                                "z0": z0,
                                "boundary": boundary,
                                "best_risk": result.best_value,
                                "risk_ratio_to_z0": result.best_value / z0,
                                "safe": int(result.best_value <= boundary),
                                "evaluations": result.evaluations,
                                "best_alpha": result.best_alpha.tolist(),
                            }
                            append_jsonl(cem_path, row)
                            existing_cem.add(key)
                            print(
                                f"[Recoverability CEM] {pde}:{ic_idx} A={authority:g} B={n_blocks} "
                                f"best={result.best_value / z0:.4f} safe={row['safe']}",
                                flush=True,
                            )

    selection_path.write_text(
        json.dumps(
            {
                "selection": "nearest K=5 HF margins, balanced across signs when available",
                "seed": args.validation_seed,
                "pool_per_pde": args.validation_pool,
                "selected_per_pde": args.validation_per_pde,
                "states": validation_selection,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    surface_rows = list(iter_jsonl(surface_path))
    cem_rows = list(iter_jsonl(cem_path))
    payload = {
        "experiment": "Finite-horizon recoverability surface",
        "interpretation": (
            "Observed recoverability is conditional on horizon, actuator authority, searched control "
            "family and finite search budget. HF-CEM is an offline diagnostic upper bound."
        ),
        "boundary_scale": args.boundary_scale,
        "surface": summarize(surface_rows)["surface"],
        "cem": summarize(
            [
                {
                    **row,
                    "family": f"HF-CEM-B{row['n_blocks']}",
                }
                for row in cem_rows
            ]
        )["surface"]
        if cem_rows
        else [],
        "n_surface_rows": len(surface_rows),
        "n_cem_rows": len(cem_rows),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    (args.outdir / "recoverability_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(args.outdir / "recoverability_summary.json"), **payload}, indent=2))


if __name__ == "__main__":
    main()

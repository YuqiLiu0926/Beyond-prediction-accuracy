#!/usr/bin/env python3
"""Gray--Scott closed-loop study with verified-continuation preservation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from continuation_library import candidate_library
from gray_scott_reference_dynamics import (
    iter_jsonl,
    make_evaluator,
    physics_rollout,
    risk_state,
    surrogate_plan,
)


BASE = Path(__file__).resolve().parents[1]
MODES = ("raw", "continuation_preserving", "continuation_only")


def physics_certificate(
    evaluator,
    state: np.ndarray,
    controls: list[np.ndarray],
    boundary: float,
    coarse_substeps: int,
    medium_substeps: int,
    coarse_allowance: float,
    medium_allowance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select and certify one control without evaluating HF dynamics."""
    if not controls:
        raise ValueError("controls must be non-empty")
    z0 = risk_state(evaluator, state)
    started = time.perf_counter()
    coarse_values = []
    for control in controls:
        _, risk = physics_rollout(evaluator, state, control, "coarse", coarse_substeps)
        coarse_values.append(float(np.max(risk)))
    coarse_seconds = time.perf_counter() - started
    coarse_upper = np.asarray(coarse_values, dtype=np.float64) + coarse_allowance * z0
    coarse_selected = int(np.argmin(coarse_upper))
    if float(coarse_upper[coarse_selected]) <= boundary:
        return np.asarray(controls[coarse_selected], dtype=np.float64), {
            "certified": 1,
            "level": "N0",
            "selected": coarse_selected,
            "state_risk": z0,
            "z_physics": float(coarse_values[coarse_selected]),
            "z_upper": float(coarse_upper[coarse_selected]),
            "N0_seconds": coarse_seconds,
            "N1_seconds": 0.0,
            "N1_evaluated": 0,
        }

    started = time.perf_counter()
    medium_values = []
    for control in controls:
        _, risk = physics_rollout(evaluator, state, control, "coarse", medium_substeps)
        medium_values.append(float(np.max(risk)))
    medium_seconds = time.perf_counter() - started
    medium_upper = np.asarray(medium_values, dtype=np.float64) + medium_allowance * z0
    medium_selected = int(np.argmin(medium_upper))
    return np.asarray(controls[medium_selected], dtype=np.float64), {
        "certified": int(float(medium_upper[medium_selected]) <= boundary),
        "level": "N1" if float(medium_upper[medium_selected]) <= boundary else "unavailable",
        "selected": medium_selected,
        "state_risk": z0,
        "z_physics": float(medium_values[medium_selected]),
        "z_upper": float(medium_upper[medium_selected]),
        "N0_best_upper": float(coarse_upper[coarse_selected]),
        "N0_seconds": coarse_seconds,
        "N1_seconds": medium_seconds,
        "N1_evaluated": 1,
    }


def composite_family(candidate: np.ndarray, continuations: np.ndarray, apply_block: int) -> list[np.ndarray]:
    prefix = np.asarray(candidate[:apply_block], dtype=np.float64)
    return [
        np.concatenate([prefix, np.asarray(continuation[apply_block:], dtype=np.float64)])
        for continuation in continuations
    ]


def run_episode(evaluator, initial_state: np.ndarray, mode: str, args, continuations: np.ndarray) -> dict[str, Any]:
    state = np.asarray(initial_state, dtype=np.float64).copy()
    initial_risk = risk_state(evaluator, state)
    boundary = float(args.boundary_scale) * initial_risk
    risks = [initial_risk]
    controls: list[float] = []
    decisions = []
    retained_continuation: np.ndarray | None = None
    retained_certificate: dict[str, Any] | None = None
    intervention_count = 0
    unavailable_count = 0
    candidate_release_count = 0
    n1_evaluations = 0
    planning_seconds = 0.0
    certificate_seconds = 0.0

    for step in range(0, int(args.episode_steps), int(args.apply_block)):
        block_len = min(int(args.apply_block), int(args.episode_steps) - step)
        remaining_episode = int(args.episode_steps) - step
        current_continuation_info = None

        if mode in {"continuation_preserving", "continuation_only"} and (
            retained_continuation is None or mode == "continuation_only"
        ):
            selected_continuation, current_continuation_info = physics_certificate(
                evaluator,
                state,
                [np.asarray(control, dtype=np.float64) for control in continuations],
                boundary,
                args.coarse_substeps,
                args.medium_substeps,
                args.coarse_allowance,
                args.medium_allowance,
            )
            certificate_seconds += current_continuation_info["N0_seconds"] + current_continuation_info["N1_seconds"]
            n1_evaluations += int(current_continuation_info["N1_evaluated"])
            if mode == "continuation_preserving" and int(current_continuation_info["certified"]):
                retained_continuation = selected_continuation.copy()
                retained_certificate = dict(current_continuation_info)

        candidate = None
        plan_info: dict[str, Any] = {}
        candidate_certificate = None
        selected_composite = None
        if mode != "continuation_only":
            started = time.perf_counter()
            candidate, plan_info = surrogate_plan(evaluator, state, boundary)
            planning_seconds += time.perf_counter() - started

        if mode == "raw":
            applied = np.asarray(candidate[:block_len], dtype=np.float64)
            source = "surrogate_candidate"
            candidate_release_count += 1
        elif mode == "continuation_only":
            if current_continuation_info is None:
                raise RuntimeError("K5 mode did not evaluate a current continuation")
            applied = np.asarray(selected_continuation[:block_len], dtype=np.float64)
            source = "continuation_only"
            unavailable_count += int(not current_continuation_info["certified"])
        else:
            composites = composite_family(candidate, continuations, int(args.apply_block))
            selected_composite, candidate_certificate = physics_certificate(
                evaluator,
                state,
                composites,
                boundary,
                args.coarse_substeps,
                args.medium_substeps,
                args.coarse_allowance,
                args.medium_allowance,
            )
            certificate_seconds += (
                candidate_certificate["N0_seconds"] + candidate_certificate["N1_seconds"]
            )
            n1_evaluations += int(candidate_certificate["N1_evaluated"])
            if int(candidate_certificate["certified"]):
                applied = np.asarray(candidate[:block_len], dtype=np.float64)
                retained_continuation = np.asarray(
                    selected_composite[int(args.apply_block) :], dtype=np.float64
                )
                retained_certificate = dict(candidate_certificate)
                source = "surrogate_candidate"
                candidate_release_count += 1
            elif retained_continuation is not None and len(retained_continuation) >= block_len:
                applied = np.asarray(retained_continuation[:block_len], dtype=np.float64)
                retained_continuation = np.asarray(
                    retained_continuation[block_len:], dtype=np.float64
                )
                source = "retained_continuation"
                intervention_count += 1
            else:
                selected_continuation, emergency = physics_certificate(
                    evaluator,
                    state,
                    [np.asarray(control, dtype=np.float64) for control in continuations],
                    boundary,
                    args.coarse_substeps,
                    args.medium_substeps,
                    args.coarse_allowance,
                    args.medium_allowance,
                )
                certificate_seconds += emergency["N0_seconds"] + emergency["N1_seconds"]
                n1_evaluations += int(emergency["N1_evaluated"])
                applied = np.asarray(selected_continuation[:block_len], dtype=np.float64)
                source = "uncertified_emergency_continuation"
                intervention_count += 1
                unavailable_count += 1
                current_continuation_info = emergency

        if mode == "continuation_preserving" and source == "retained_continuation":
            if retained_certificate is None or not int(retained_certificate["certified"]):
                raise RuntimeError("A retained continuation was used without a valid certificate")
        if mode == "continuation_preserving" and source == "surrogate_candidate":
            if retained_continuation is None or len(retained_continuation) < max(
                0, remaining_episode - block_len
            ):
                raise RuntimeError("Accepted candidate did not retain a complete verified continuation")

        hf_states, hf_risk = physics_rollout(evaluator, state, applied, "hf", args.coarse_substeps)
        state = hf_states[-1]
        risks.extend(float(value) for value in hf_risk[1:])
        controls.extend(float(value) for value in applied)
        decisions.append(
            {
                "step": step,
                "source": source,
                "current_risk": float(risks[-block_len - 1]),
                "current_continuation": current_continuation_info,
                "candidate_certificate": candidate_certificate,
                "retained_certificate_level": (
                    retained_certificate["level"] if retained_certificate else None
                ),
                "retained_length_after": (
                    int(len(retained_continuation)) if retained_continuation is not None else 0
                ),
                "plan_info": plan_info,
                "block_max_hf_risk": float(np.max(hf_risk)),
            }
        )

    risk_array = np.asarray(risks, dtype=np.float64)
    violations = risk_array > boundary
    states_full, replay_risk = physics_rollout(
        evaluator, initial_state, np.asarray(controls, dtype=np.float64), "hf", args.coarse_substeps
    )
    del states_full
    replay_error = float(np.max(np.abs(replay_risk - risk_array)))
    if replay_error > 1e-8:
        raise RuntimeError(f"Episode HF replay mismatch: {replay_error}")
    return {
        "mode": mode,
        "initial_risk": initial_risk,
        "boundary": boundary,
        "episode_steps": int(args.episode_steps),
        "apply_block": int(args.apply_block),
        "max_hf_risk": float(np.max(risk_array)),
        "max_ratio_to_boundary": float(np.max(risk_array) / boundary),
        "unsafe": int(np.any(violations)),
        "violation_steps": int(np.sum(violations)),
        "first_violation_step": int(np.flatnonzero(violations)[0]) if np.any(violations) else None,
        "interventions": intervention_count,
        "unavailable_decisions": unavailable_count,
        "candidate_releases": candidate_release_count,
        "N1_evaluations": n1_evaluations,
        "planning_seconds": planning_seconds,
        "certificate_seconds": certificate_seconds,
        "hf_replay_max_risk_error": replay_error,
        "risk_trace": risk_array.tolist(),
        "controls": controls,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE / "external_data/decision_archives/evaluation")
    parser.add_argument("--outdir", type=Path, default=BASE / "outputs/closed_loop/gray_scott_development")
    parser.add_argument("--model", default="fno")
    parser.add_argument("--rho", type=float, default=0.90)
    parser.add_argument("--indices", default="19,27,43,61,62,79,91")
    parser.add_argument("--max-ics", type=int, default=0)
    parser.add_argument("--modes", default="raw,continuation_preserving,continuation_only")
    parser.add_argument("--planning-horizon", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--apply-block", type=int, default=10)
    parser.add_argument("--boundary-scale", type=float, default=1.20)
    parser.add_argument("--coarse-substeps", type=int, default=1)
    parser.add_argument("--medium-substeps", type=int, default=2)
    parser.add_argument("--coarse-allowance", type=float, default=0.00580307067985165)
    parser.add_argument("--medium-allowance", type=float, default=0.003975447346840485)
    parser.add_argument("--ic-seed", type=int, default=20260731)
    parser.add_argument("--n-ics", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--ckpt-root", type=Path, default=BASE / "external_data/checkpoints/discovery")
    parser.add_argument("--model-data-root", type=Path, default=BASE / "external_data/datasets/discovery")
    parser.add_argument("--train-script", type=Path, default=BASE / "scripts/train_discovery_surrogates.py")
    parser.add_argument("--burgers-script", type=Path, default=BASE / "scripts/generate_burgers_decisions.py")
    parser.add_argument("--grayscott-script", type=Path, default=BASE / "scripts/generate_gray_scott_decisions.py")
    parser.add_argument("--kolmogorov-script", type=Path, default=BASE / "scripts/generate_kolmogorov_decisions.py")
    args = parser.parse_args()

    if args.planning_horizon < args.episode_steps:
        raise ValueError("planning_horizon must cover the finite confirmation episode")
    if args.episode_steps % args.apply_block:
        raise ValueError("episode_steps must be divisible by apply_block")
    if not 1 <= args.coarse_substeps < args.medium_substeps:
        raise ValueError("Require coarse_substeps < medium_substeps")
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    if any(mode not in MODES for mode in modes):
        raise ValueError(args.modes)
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    if args.max_ics > 0:
        indices = indices[: args.max_ics]
    if not indices or min(indices) < 0 or max(indices) >= args.n_ics:
        raise ValueError(f"Invalid indices for n_ics={args.n_ics}: {indices}")

    evaluator = make_evaluator(args)
    ood_states = np.asarray(evaluator.x0s, dtype=np.float64)
    evaluator.refresh_ics(args.n_ics, "high_gradient")
    base_states = np.asarray(evaluator.x0s, dtype=np.float64)
    names, library = candidate_library(args.planning_horizon, evaluator.umin, evaluator.umax)
    continuations = np.asarray(library[:5], dtype=np.float64)

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows_path = args.outdir / "episode_rows.jsonl"
    if not int(args.resume) and rows_path.exists():
        rows_path.unlink()
    existing = list(iter_jsonl(rows_path))
    done = {(int(row["ic_idx"]), str(row["mode"])) for row in existing}
    started = time.perf_counter()
    for ic_idx in indices:
        initial = base_states[ic_idx] + float(args.rho) * (ood_states[ic_idx] - base_states[ic_idx])
        for mode in modes:
            if (ic_idx, mode) in done:
                continue
            episode = run_episode(evaluator, initial, mode, args, continuations)
            row = {
                "pde": "grayscott",
                "model": args.model,
                "ic_seed": args.ic_seed,
                "ic_idx": ic_idx,
                "rho": args.rho,
                "continuation_names": names[:5],
                **episode,
            }
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"[Gray--Scott closed-loop study] ic={ic_idx} mode={mode} unsafe={row['unsafe']} "
                f"max/bound={row['max_ratio_to_boundary']:.4f} release={row['candidate_releases']} "
                f"interventions={row['interventions']}",
                flush=True,
            )

    all_rows = list(iter_jsonl(rows_path))
    expected = {(ic_idx, mode) for ic_idx in indices for mode in modes}
    selected_rows = [
        row for row in all_rows if (int(row["ic_idx"]), str(row["mode"])) in expected
    ]
    keys = {(int(row["ic_idx"]), str(row["mode"])) for row in selected_rows}
    if keys != expected or len(selected_rows) != len(expected):
        raise RuntimeError(f"Incomplete or duplicate episodes: {len(selected_rows)}/{len(expected)}")

    by_mode = {}
    for mode in modes:
        local = [row for row in selected_rows if row["mode"] == mode]
        total_decisions = sum(len(row["decisions"]) for row in local)
        by_mode[mode] = {
            "episodes": len(local),
            "unsafe_episodes": sum(int(row["unsafe"]) for row in local),
            "unavailable_decisions": sum(int(row["unavailable_decisions"]) for row in local),
            "candidate_release_fraction": float(
                sum(int(row["candidate_releases"]) for row in local) / total_decisions
            ),
            "interventions": sum(int(row["interventions"]) for row in local),
            "N1_evaluations": sum(int(row["N1_evaluations"]) for row in local),
            "mean_max_ratio_to_boundary": float(
                np.mean([float(row["max_ratio_to_boundary"]) for row in local])
            ),
            "mean_planning_seconds_per_decision": float(
                sum(float(row["planning_seconds"]) for row in local) / total_decisions
            ),
            "mean_certificate_seconds_per_decision": float(
                sum(float(row["certificate_seconds"]) for row in local) / total_decisions
            ),
        }
    summary = {
        "status": "DEVELOPMENT_ANALYSIS",
        "method": "continuation_preserving_controller",
        "pde": "grayscott",
        "model": args.model,
        "ic_seed": args.ic_seed,
        "indices": indices,
        "rho": args.rho,
        "planning_horizon": args.planning_horizon,
        "episode_steps": args.episode_steps,
        "apply_block": args.apply_block,
        "boundary_scale": args.boundary_scale,
        "N0_substeps": args.coarse_substeps,
        "N1_substeps": args.medium_substeps,
        "N0_allowance": args.coarse_allowance,
        "N1_allowance": args.medium_allowance,
        "by_mode": by_mode,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": str(rows_path),
    }
    summary_path = args.outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "by_mode": by_mode}, indent=2))


if __name__ == "__main__":
    main()

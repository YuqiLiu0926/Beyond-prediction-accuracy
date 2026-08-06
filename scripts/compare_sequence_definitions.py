#!/usr/bin/env python3
"""Causal Gray-Scott ablations for certificate-object alignment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from continuation_library import candidate_library
from gray_scott_reference_dynamics import (
    make_evaluator,
    physics_rollout,
    risk_state,
    surrogate_plan,
)
from gray_scott_continuation_controller import physics_certificate


BASE = Path(__file__).resolve().parents[1]
MODES = ("full_plan", "block_only")


def run_episode(
    evaluator,
    initial_state: np.ndarray,
    mode: str,
    args,
    continuations: np.ndarray,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(mode)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    initial_risk = risk_state(evaluator, state)
    boundary = float(args.boundary_scale) * initial_risk
    risks = [initial_risk]
    controls: list[float] = []
    decisions: list[dict[str, Any]] = []
    retained_continuation: np.ndarray | None = None
    retained_certificate: dict[str, Any] | None = None
    interventions = 0
    unavailable = 0
    candidate_releases = 0
    n1_evaluations = 0
    planning_seconds = 0.0
    certificate_seconds = 0.0

    for step in range(0, int(args.episode_steps), int(args.apply_block)):
        block_len = min(int(args.apply_block), int(args.episode_steps) - step)
        remaining_episode = int(args.episode_steps) - step
        current_continuation = None
        selected_continuation = None

        if retained_continuation is None or mode == "block_only":
            selected_continuation, current_continuation = physics_certificate(
                evaluator,
                state,
                [np.asarray(control, dtype=np.float64) for control in continuations],
                boundary,
                args.coarse_substeps,
                args.medium_substeps,
                args.coarse_allowance,
                args.medium_allowance,
            )
            certificate_seconds += current_continuation["N0_seconds"] + current_continuation["N1_seconds"]
            n1_evaluations += int(current_continuation["N1_evaluated"])
            if mode == "full_plan" and int(current_continuation["certified"]):
                retained_continuation = np.asarray(selected_continuation, dtype=np.float64).copy()
                retained_certificate = dict(current_continuation)

        started = time.perf_counter()
        candidate, plan_info = surrogate_plan(evaluator, state, boundary)
        planning_seconds += time.perf_counter() - started
        certificate_control = (
            np.asarray(candidate, dtype=np.float64)
            if mode == "full_plan"
            else np.asarray(candidate[:block_len], dtype=np.float64)
        )
        _, candidate_certificate = physics_certificate(
            evaluator,
            state,
            [certificate_control],
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
            source = "surrogate_candidate"
            candidate_releases += 1
            if mode == "full_plan":
                retained_continuation = np.asarray(candidate[block_len:], dtype=np.float64)
                retained_certificate = dict(candidate_certificate)
                if len(retained_continuation) < max(0, remaining_episode - block_len):
                    raise RuntimeError("Full-plan evaluation retained an incomplete continuation")
            else:
                # Block-only evaluation provides no verified continuation from the
                # reached state, so the previously retained sequence is invalid.
                retained_continuation = None
                retained_certificate = None
        elif retained_continuation is not None and len(retained_continuation) >= block_len:
            applied = np.asarray(retained_continuation[:block_len], dtype=np.float64)
            retained_continuation = np.asarray(
                retained_continuation[block_len:], dtype=np.float64
            )
            source = "retained_continuation"
            interventions += 1
        elif selected_continuation is not None:
            applied = np.asarray(selected_continuation[:block_len], dtype=np.float64)
            source = "current_continuation" if int(current_continuation["certified"]) else "uncertified_continuation"
            interventions += 1
            unavailable += int(not current_continuation["certified"])
        else:
            raise RuntimeError("No executable ablation action")

        if source == "retained_continuation" and (
            retained_certificate is None or not int(retained_certificate["certified"])
        ):
            raise RuntimeError("Uncertified retained continuation")
        hf_states, hf_risk = physics_rollout(
            evaluator, state, applied, "hf", args.coarse_substeps
        )
        state = hf_states[-1]
        controls.extend(float(value) for value in applied)
        risks.extend(float(value) for value in hf_risk[1:])
        decisions.append(
            {
                "step": step,
                "source": source,
                "current_continuation": current_continuation,
                "candidate_certificate": candidate_certificate,
                "retained_certificate_level": (
                    retained_certificate["level"] if retained_certificate else None
                ),
                "retained_length_after": (
                    len(retained_continuation) if retained_continuation is not None else 0
                ),
                "plan_info": plan_info,
                "block_max_hf_risk": float(np.max(hf_risk)),
            }
        )

    risk_array = np.asarray(risks, dtype=np.float64)
    violation = risk_array > boundary
    _, replay_risk = physics_rollout(
        evaluator,
        initial_state,
        np.asarray(controls, dtype=np.float64),
        "hf",
        args.coarse_substeps,
    )
    replay_error = float(np.max(np.abs(replay_risk - risk_array)))
    if replay_error > 1e-8:
        raise RuntimeError(f"HF replay mismatch: {replay_error}")
    return {
        "mode": mode,
        "initial_risk": initial_risk,
        "boundary": boundary,
        "episode_steps": int(args.episode_steps),
        "apply_block": int(args.apply_block),
        "max_hf_risk": float(np.max(risk_array)),
        "max_ratio_to_boundary": float(np.max(risk_array) / boundary),
        "unsafe": int(np.any(violation)),
        "violation_steps": int(np.sum(violation)),
        "first_violation_step": int(np.flatnonzero(violation)[0]) if np.any(violation) else None,
        "interventions": interventions,
        "unavailable_decisions": unavailable,
        "candidate_releases": candidate_releases,
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
    parser.add_argument(
        "--data-root",
        type=Path,
        default=BASE / "external_data/decision_archives/evaluation",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=BASE / "outputs/analysis/sequence_definition_ablation",
    )
    parser.add_argument("--model", default="fno")
    parser.add_argument("--rho", type=float, default=0.90)
    parser.add_argument(
        "--indices",
        default="82,65,14,96,66,18,52,86,8,73,20,98,42,0,21,56",
    )
    parser.add_argument("--ic-seed", type=int, default=20260811)
    parser.add_argument("--n-ics", type=int, default=100)
    parser.add_argument("--planning-horizon", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--apply-block", type=int, default=10)
    parser.add_argument("--boundary-scale", type=float, default=1.20)
    parser.add_argument("--coarse-substeps", type=int, default=1)
    parser.add_argument("--medium-substeps", type=int, default=2)
    parser.add_argument("--coarse-allowance", type=float, default=0.00580307067985165)
    parser.add_argument("--medium-allowance", type=float, default=0.003975447346840485)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=BASE / "external_data/checkpoints/discovery",
    )
    parser.add_argument(
        "--model-data-root",
        type=Path,
        default=BASE / "external_data/datasets/discovery",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=BASE / "scripts/train_discovery_surrogates.py",
    )
    parser.add_argument(
        "--burgers-script",
        type=Path,
        default=BASE / "scripts/generate_burgers_decisions.py",
    )
    parser.add_argument(
        "--grayscott-script",
        type=Path,
        default=BASE / "scripts/generate_gray_scott_decisions.py",
    )
    parser.add_argument(
        "--kolmogorov-script",
        type=Path,
        default=BASE / "scripts/generate_kolmogorov_decisions.py",
    )
    args = parser.parse_args()
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if any(mode not in MODES for mode in modes):
        raise ValueError(modes)
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    evaluator = make_evaluator(args)
    ood = np.asarray(evaluator.x0s, dtype=np.float64)
    evaluator.refresh_ics(args.n_ics, "high_gradient")
    base = np.asarray(evaluator.x0s, dtype=np.float64)
    states = base + float(args.rho) * (ood - base)
    names, continuations = candidate_library(
        args.planning_horizon, evaluator.umin, evaluator.umax
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows_path = args.outdir / "episode_rows.jsonl"
    existing: set[tuple[int, str]] = set()
    if args.resume and rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing.add((int(row["ic_idx"]), str(row["mode"])))
    with rows_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for ic_idx in indices:
            for mode in modes:
                if (ic_idx, mode) in existing:
                    continue
                row = run_episode(evaluator, states[ic_idx], mode, args, continuations)
                row.update(
                    {
                        "pde": "grayscott",
                        "model": args.model,
                        "ic_seed": args.ic_seed,
                        "ic_idx": ic_idx,
                        "rho": args.rho,
                        "continuation_names": names,
                    }
                )
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                print(
                    f"[sequence definition] ic={ic_idx} mode={mode} unsafe={row['unsafe']} "
                    f"release={row['candidate_releases']} unavailable={row['unavailable_decisions']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()

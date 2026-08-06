#!/usr/bin/env python3
"""Gray--Scott reference dynamics for controller development and calibration."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from continuation_library import candidate_library
from surrogate_pair_evaluator import Config, PairEvaluator


BASE = Path(__file__).resolve().parents[1]
MODES = ("raw", "continuation_preserving", "continuation_only")


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def make_evaluator(args: argparse.Namespace) -> PairEvaluator:
    cfg = Config()
    cfg.root = str(args.data_root)
    cfg.ckpt_root = str(args.ckpt_root)
    cfg.data_root = str(args.model_data_root)
    cfg.train_script = str(args.train_script)
    cfg.burgers_script = str(args.burgers_script)
    cfg.grayscott_script = str(args.grayscott_script)
    cfg.kolmogorov_script = str(args.kolmogorov_script)
    cfg.ic_mode = "ood_amplified"
    cfg.H = int(args.planning_horizon)
    cfg.canonical_horizon = int(args.planning_horizon)
    cfg.device = str(args.device)
    evaluator = PairEvaluator("grayscott", args.model, cfg)
    evaluator.exp_cfg.seed = int(args.ic_seed)
    evaluator.refresh_ics(args.n_ics, "ood_amplified")
    return evaluator


def risk_state(evaluator: PairEvaluator, state: np.ndarray) -> float:
    return float(evaluator.mod.initial_front_risk(state, evaluator.dx, evaluator.dy))


def physics_rollout(
    evaluator: PairEvaluator, state: np.ndarray, control: np.ndarray, fidelity: str, coarse_substeps: int
) -> tuple[np.ndarray, np.ndarray]:
    dt_true = (
        float(evaluator.exp_cfg.dt_true)
        if fidelity == "hf"
        else float(evaluator.dt_nom) / int(coarse_substeps)
    )
    states = evaluator.mod.rollout_true(
        s0_true=np.asarray(state, dtype=np.float64),
        a_seq=np.asarray(control, dtype=np.float64),
        actuator=evaluator.actuator,
        dx=evaluator.dx,
        dy=evaluator.dy,
        Du=evaluator.Du,
        Dv=evaluator.Dv,
        F0=evaluator.F0,
        k0=evaluator.k0,
        dt_nom=evaluator.dt_nom,
        dt_true=dt_true,
    )
    series = np.asarray([risk_state(evaluator, value) for value in states], dtype=np.float64)
    return np.asarray(states, dtype=np.float64), series


def continuation_only_recovery(
    evaluator: PairEvaluator,
    state: np.ndarray,
    boundary: float,
    candidates: np.ndarray,
    coarse_substeps: int,
    normalized_allowance: float,
) -> dict[str, Any]:
    values = []
    for control in candidates:
        _, series = physics_rollout(evaluator, state, control, "coarse", coarse_substeps)
        values.append(float(np.max(series)))
    selected = int(np.argmin(values))
    z0 = risk_state(evaluator, state)
    upper = float(values[selected] + normalized_allowance * z0)
    return {
        "selected": selected,
        "z_lf": float(values[selected]),
        "z_upper": upper,
        "certified": int(upper <= boundary),
        "state_risk": z0,
    }


def surrogate_plan(evaluator: PairEvaluator, state: np.ndarray, boundary: float) -> tuple[np.ndarray, dict[str, Any]]:
    alpha, control, info = evaluator.mod.sequential_linearized_programming(
        s0=np.asarray(state, dtype=np.float64),
        model=evaluator.model,
        model_type=evaluator.model_type,
        coords_2d=evaluator.coords_2d,
        actuator=evaluator.actuator,
        dx=evaluator.dx,
        dy=evaluator.dy,
        cfg=evaluator.exp_cfg,
        z_limit_override=float(boundary),
    )
    return np.asarray(control, dtype=np.float64), {
        "alpha": np.asarray(alpha, dtype=np.float64).tolist(),
        "planner": {key: float(value) if isinstance(value, (float, np.floating)) else value for key, value in info.items()},
    }


def run_episode(
    evaluator: PairEvaluator,
    initial_state: np.ndarray,
    mode: str,
    args: argparse.Namespace,
    candidates: np.ndarray,
) -> dict[str, Any]:
    state = np.asarray(initial_state, dtype=np.float64).copy()
    initial_risk = risk_state(evaluator, state)
    boundary = float(args.boundary_scale) * initial_risk
    risks = [initial_risk]
    controls: list[float] = []
    decisions = []
    intervention_count = 0
    unavailable_count = 0
    planning_seconds = 0.0
    physics_gate_seconds = 0.0

    for step in range(0, int(args.episode_steps), int(args.apply_block)):
        block_len = min(int(args.apply_block), int(args.episode_steps) - step)
        gate_started = time.perf_counter()
        current_recovery = continuation_only_recovery(
            evaluator,
            state,
            boundary,
            candidates,
            args.coarse_substeps,
            args.normalized_coarse_allowance,
        )
        physics_gate_seconds += time.perf_counter() - gate_started

        candidate = None
        plan_info: dict[str, Any] = {}
        if mode != "continuation_only":
            plan_started = time.perf_counter()
            candidate, plan_info = surrogate_plan(evaluator, state, boundary)
            planning_seconds += time.perf_counter() - plan_started

        candidate_path_upper = None
        next_recovery = None
        accepted_candidate = False
        if mode in {"raw", "continuation_preserving"}:
            candidate_block = np.asarray(candidate[:block_len], dtype=np.float64)
            gate_started = time.perf_counter()
            coarse_full_states, coarse_full_risk = physics_rollout(
                evaluator, state, candidate, "coarse", args.coarse_substeps
            )
            candidate_path_upper = float(
                np.max(coarse_full_risk) + args.normalized_coarse_allowance * current_recovery["state_risk"]
            )
            coarse_block_states, _ = physics_rollout(
                evaluator, state, candidate_block, "coarse", args.coarse_substeps
            )
            next_recovery = continuation_only_recovery(
                evaluator,
                coarse_block_states[-1],
                boundary,
                candidates,
                args.coarse_substeps,
                args.normalized_coarse_allowance,
            )
            physics_gate_seconds += time.perf_counter() - gate_started
            accepted_candidate = bool(
                mode == "raw"
                or (candidate_path_upper <= boundary and int(next_recovery["certified"]) == 1)
            )

        if mode == "continuation_only" or not accepted_candidate:
            selected = int(current_recovery["selected"])
            applied = np.asarray(candidates[selected][:block_len], dtype=np.float64)
            intervention_count += int(mode == "continuation_preserving")
            unavailable_count += int(not current_recovery["certified"])
            source = "continuation_only"
        else:
            applied = np.asarray(candidate[:block_len], dtype=np.float64)
            source = "surrogate_candidate"

        hf_states, hf_risk = physics_rollout(evaluator, state, applied, "hf", args.coarse_substeps)
        state = hf_states[-1]
        risks.extend(float(value) for value in hf_risk[1:])
        controls.extend(float(value) for value in applied)
        decisions.append(
            {
                "step": step,
                "source": source,
                "current_risk": float(risks[-block_len - 1]),
                "current_continuation": current_recovery,
                "candidate_path_upper": candidate_path_upper,
                "next_continuation": next_recovery,
                "plan_info": plan_info,
                "block_max_hf_risk": float(np.max(hf_risk)),
            }
        )

    risk_array = np.asarray(risks, dtype=np.float64)
    violations = risk_array > boundary
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
        "planning_seconds": planning_seconds,
        "physics_gate_seconds": physics_gate_seconds,
        "risk_trace": risk_array.tolist(),
        "controls": controls,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE / "external_data/decision_archives/evaluation")
    parser.add_argument("--oracle-rows", type=Path, default=BASE / "external_data/analysis/recoverability/reference_search_rows.jsonl")
    parser.add_argument("--outdir", type=Path, default=BASE / "outputs/calibration/gray_scott")
    parser.add_argument("--model", default="fno")
    parser.add_argument("--rho", type=float, default=0.90)
    parser.add_argument("--max-ics", type=int, default=7)
    parser.add_argument("--modes", default="raw,continuation_preserving,continuation_only")
    parser.add_argument("--planning-horizon", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--apply-block", type=int, default=10)
    parser.add_argument("--boundary-scale", type=float, default=1.20)
    parser.add_argument("--coarse-substeps", type=int, default=2)
    parser.add_argument("--normalized-coarse-allowance", type=float, default=0.003975447346840485)
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
    if args.apply_block < 1 or args.episode_steps % args.apply_block != 0:
        raise ValueError("episode_steps must be divisible by apply_block")
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    if any(mode not in MODES for mode in modes):
        raise ValueError(args.modes)

    evaluator = make_evaluator(args)
    ood_states = np.asarray(evaluator.x0s, dtype=np.float64)
    evaluator.refresh_ics(args.n_ics, "high_gradient")
    base_states = np.asarray(evaluator.x0s, dtype=np.float64)
    source = [row for row in iter_jsonl(args.oracle_rows) if row["pde"] == "grayscott" and not int(row["continuation_only_hf_safe"])]
    indices = [int(row["ic_idx"]) for row in source][: int(args.max_ics)]
    names, library = candidate_library(args.planning_horizon, evaluator.umin, evaluator.umax)
    candidates = np.asarray(library[:5], dtype=np.float64)

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
            episode = run_episode(evaluator, initial, mode, args, candidates)
            row = {
                "pde": "grayscott",
                "model": args.model,
                "ic_idx": ic_idx,
                "rho": args.rho,
                "candidate_names": names[:5],
                **episode,
            }
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"[Gray--Scott calibration study] ic={ic_idx} mode={mode} unsafe={row['unsafe']} "
                f"max/bound={row['max_ratio_to_boundary']:.4f} interventions={row['interventions']}",
                flush=True,
            )

    rows = list(iter_jsonl(rows_path))
    expected = {(ic_idx, mode) for ic_idx in indices for mode in modes}
    selected = [row for row in rows if (int(row["ic_idx"]), str(row["mode"])) in expected]
    keys = {(int(row["ic_idx"]), str(row["mode"])) for row in selected}
    if keys != expected or len(selected) != len(expected):
        raise RuntimeError(f"Incomplete episodes {len(selected)}/{len(expected)}")
    by_mode = {}
    for mode in modes:
        local = [row for row in selected if row["mode"] == mode]
        by_mode[mode] = {
            "episodes": len(local),
            "unsafe_episodes": sum(int(row["unsafe"]) for row in local),
            "mean_max_ratio_to_boundary": float(np.mean([row["max_ratio_to_boundary"] for row in local])),
            "total_interventions": sum(int(row["interventions"]) for row in local),
            "total_unavailable_decisions": sum(int(row["unavailable_decisions"]) for row in local),
            "mean_planning_seconds": float(np.mean([row["planning_seconds"] for row in local])),
            "mean_physics_gate_seconds": float(np.mean([row["physics_gate_seconds"] for row in local])),
        }
    summary = {
        "status": "CALIBRATION_ANALYSIS",
        "pde": "grayscott",
        "model": args.model,
        "rho": args.rho,
        "indices": indices,
        "planning_horizon": args.planning_horizon,
        "episode_steps": args.episode_steps,
        "apply_block": args.apply_block,
        "fixed_boundary_scale": args.boundary_scale,
        "normalized_coarse_allowance": args.normalized_coarse_allowance,
        "by_mode": by_mode,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": str(rows_path),
    }
    summary_path = args.outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary": str(summary_path), "by_mode": by_mode}, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate static initial-condition design as a PDE decision task outside feedback control.

For each independently generated Gray--Scott state, a fixed bank of localized
initial-condition perturbations is evaluated by four frozen neural surrogates.
Each surrogate selects the predicted-best design among those below a common
state-level risk limit. HF trajectories are used only after selection and for
an offline oracle comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

from gray_scott_reference_dynamics import make_evaluator, physics_rollout, risk_state


BASE = Path(__file__).resolve().parents[1]
MODELS = ("deeponet", "fno", "pino", "pinn")


def state_hash(state: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(state, dtype=np.float32).tobytes()).hexdigest()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def periodic_delta(values: np.ndarray, center: float, length: float) -> np.ndarray:
    delta = np.abs(values - center)
    return np.minimum(delta, length - delta)


def design_bank(
    state: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float,
    Ly: float,
    sigma_fraction: float,
) -> tuple[list[str], np.ndarray]:
    """Create one unchanged state and fifteen predefined localized designs."""
    centers = ((0.20, 0.25), (0.35, 0.70), (0.50, 0.45), (0.70, 0.75), (0.82, 0.30))
    amplitudes = (-0.06, 0.04, 0.08)
    sigma = sigma_fraction * min(Lx, Ly)
    names = ["unchanged"]
    candidates = [np.asarray(state, dtype=np.float64).copy()]
    for center_index, (cx_frac, cy_frac) in enumerate(centers):
        dx = periodic_delta(X, cx_frac * Lx, Lx)
        dy = periodic_delta(Y, cy_frac * Ly, Ly)
        patch = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        for amplitude in amplitudes:
            candidate = np.asarray(state, dtype=np.float64).copy()
            candidate[1] = np.clip(candidate[1] + amplitude * patch, 0.0, 1.5)
            candidate[0] = np.clip(candidate[0] - 0.5 * amplitude * patch, 0.0, 1.5)
            names.append(f"c{center_index}_a{amplitude:+.2f}")
            candidates.append(candidate)
    return names, np.stack(candidates, axis=0)


def surrogate_rollout(evaluator: Any, state: np.ndarray, horizon: int) -> dict[str, Any]:
    control = np.zeros(horizon, dtype=np.float64)
    states = np.asarray(
        evaluator.mod.surrogate_rollout(
            evaluator.model,
            evaluator.model_type,
            np.asarray(state, dtype=np.float64),
            control,
            evaluator.coords_2d,
        ),
        dtype=np.float64,
    )
    risk = float(evaluator.mod.risk_Z_front(states, evaluator.dx, evaluator.dy))
    performance = float(evaluator.mod.nominal_performance(states, evaluator.actuator))
    return {"states": states, "risk": risk, "performance": performance}


def hf_rollout(evaluator: Any, state: np.ndarray, horizon: int) -> dict[str, Any]:
    states, risk_trace = physics_rollout(
        evaluator,
        np.asarray(state, dtype=np.float64),
        np.zeros(horizon, dtype=np.float64),
        "hf",
        1,
    )
    performance = float(evaluator.mod.nominal_performance(states, evaluator.actuator))
    return {
        "states": states,
        "risk": float(np.max(risk_trace)),
        "performance": performance,
        "risk_trace": np.asarray(risk_trace, dtype=np.float64),
    }


def choose(risks: np.ndarray, performance: np.ndarray, limit: float) -> int | None:
    feasible = np.flatnonzero(risks <= limit)
    if len(feasible) == 0:
        return None
    return int(feasible[int(np.argmax(performance[feasible]))])


def grouped_bootstrap(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, list[float]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["state_idx"])].append(row)
    model_count = len({str(row["model"]) for row in rows})
    if any(len(local) != model_count for local in groups.values()):
        raise RuntimeError("Each physical state must retain every requested surrogate decision")
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        selected = rng.choice(keys, size=len(keys), replace=True)
        sample = [row for key in selected for row in groups[int(key)]]
        draws["selected_false_safe_fraction"].append(
            float(np.mean([row["selected_false_safe"] for row in sample]))
        )
        draws["exact_oracle_selection_fraction"].append(
            float(np.mean([row["selected_matches_hf_oracle"] for row in sample]))
        )
        finite_regret = [row["normalized_safe_regret"] for row in sample if row["normalized_safe_regret"] is not None]
        draws["mean_normalized_safe_regret"].append(float(np.mean(finite_regret)))
    return {
        key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for key, values in draws.items()
    }


def evaluator_args(args: argparse.Namespace, model: str) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=args.data_root,
        model=model,
        rho=0.0,
        planning_horizon=args.horizon,
        ic_seed=args.seed,
        n_ics=args.states,
        device=args.device,
        ckpt_root=args.ckpt_root,
        model_data_root=args.model_data_root,
        train_script=args.train_script,
        burgers_script=args.burgers_script,
        grayscott_script=args.grayscott_script,
        kolmogorov_script=args.kolmogorov_script,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=BASE / "outputs/static_design",
    )
    parser.add_argument("--states", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--boundary-scale", type=float, default=1.02)
    parser.add_argument("--sigma-fraction", type=float, default=0.10)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=BASE / "external_data" / "decision_archives" / "evaluation")
    parser.add_argument("--ckpt-root", type=Path, default=BASE / "external_data")
    parser.add_argument("--model-data-root", type=Path, default=BASE / "external_data")
    parser.add_argument("--train-script", type=Path, default=BASE / "scripts" / "train_discovery_surrogates.py")
    parser.add_argument("--burgers-script", type=Path, default=BASE / "scripts/generate_burgers_decisions.py")
    parser.add_argument("--grayscott-script", type=Path, default=BASE / "scripts/generate_gray_scott_decisions.py")
    parser.add_argument("--kolmogorov-script", type=Path, default=BASE / "scripts/generate_kolmogorov_decisions.py")
    args = parser.parse_args()
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    if any(model not in MODELS for model in models):
        raise ValueError(models)
    args.outdir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.outdir / "candidate_rows.jsonl"
    decision_path = args.outdir / "decision_rows.jsonl"
    if not args.resume:
        for path in (candidate_path, decision_path):
            if path.exists():
                path.unlink()

    hf_by_state: dict[int, list[dict[str, Any]]] = {}
    canonical_hashes: dict[int, str] = {}
    reference_evaluator = make_evaluator(evaluator_args(args, models[0]))
    reference_evaluator.refresh_ics(args.states, "high_gradient")
    base_states = np.asarray(reference_evaluator.x0s, dtype=np.float64)
    for state_idx, base in enumerate(base_states):
        names, candidates = design_bank(
            base,
            reference_evaluator.X,
            reference_evaluator.Y,
            reference_evaluator.Lx,
            reference_evaluator.Ly,
            args.sigma_fraction,
        )
        canonical_hashes[state_idx] = state_hash(base)
        local = []
        for candidate_idx, (name, candidate) in enumerate(zip(names, candidates)):
            hf = hf_rollout(reference_evaluator, candidate, args.horizon)
            local.append(
                {
                    "candidate_idx": candidate_idx,
                    "candidate_name": name,
                    "risk_hf": hf["risk"],
                    "performance_hf": hf["performance"],
                }
            )
        hf_by_state[state_idx] = local
        print(f"[static design HF] state={state_idx} candidates={len(local)}", flush=True)

    existing_candidates = {
        (row["model"], row["state_idx"], row["candidate_idx"]): row
        for row in iter_jsonl(candidate_path)
    }
    existing_decisions = {
        (row["model"], row["state_idx"]): row for row in iter_jsonl(decision_path)
    }
    for model in models:
        evaluator = make_evaluator(evaluator_args(args, model))
        evaluator.refresh_ics(args.states, "high_gradient")
        model_states = np.asarray(evaluator.x0s, dtype=np.float64)
        for state_idx, base in enumerate(model_states):
            if state_hash(base) != canonical_hashes[state_idx]:
                raise RuntimeError(f"Cross-model initial-state mismatch for state {state_idx}")
            names, candidates = design_bank(
                base,
                evaluator.X,
                evaluator.Y,
                evaluator.Lx,
                evaluator.Ly,
                args.sigma_fraction,
            )
            limit = args.boundary_scale * float(hf_by_state[state_idx][0]["risk_hf"])
            local_rows = []
            for candidate_idx, (name, candidate) in enumerate(zip(names, candidates)):
                key = (model, state_idx, candidate_idx)
                hf = hf_by_state[state_idx][candidate_idx]
                if key in existing_candidates:
                    row = existing_candidates[key]
                else:
                    surrogate = surrogate_rollout(evaluator, candidate, args.horizon)
                    row = {
                        "pde": "grayscott",
                        "task": "static_initial_condition_design",
                        "model": model,
                        "state_idx": state_idx,
                        "state_hash": canonical_hashes[state_idx],
                        "candidate_idx": candidate_idx,
                        "candidate_name": name,
                        "risk_limit": limit,
                        "risk_sur": surrogate["risk"],
                        "risk_hf": hf["risk_hf"],
                        "performance_sur": surrogate["performance"],
                        "performance_hf": hf["performance_hf"],
                    }
                    append_jsonl(candidate_path, row)
                    existing_candidates[key] = row
                local_rows.append(row)

            if (model, state_idx) in existing_decisions:
                continue
            risk_sur = np.asarray([row["risk_sur"] for row in local_rows], dtype=np.float64)
            risk_hf = np.asarray([row["risk_hf"] for row in local_rows], dtype=np.float64)
            task_sur = np.asarray([row["performance_sur"] for row in local_rows], dtype=np.float64)
            task_hf = np.asarray([row["performance_hf"] for row in local_rows], dtype=np.float64)
            selected = choose(risk_sur, task_sur, limit)
            oracle = choose(risk_hf, task_hf, limit)
            if selected is None or oracle is None:
                normalized_regret = None
            elif risk_hf[selected] > limit:
                normalized_regret = None
            else:
                normalized_regret = float(
                    (task_hf[oracle] - task_hf[selected]) / (abs(task_hf[oracle]) + 1e-12)
                )
            correlation = spearmanr(task_sur, task_hf).statistic
            selected_field_error = None
            if selected is not None:
                selected_surrogate = surrogate_rollout(
                    evaluator, candidates[selected], args.horizon
                )["states"]
                selected_hf = hf_rollout(
                    evaluator, candidates[selected], args.horizon
                )["states"]
                selected_field_error = float(
                    np.linalg.norm(selected_surrogate - selected_hf)
                    / (float(np.linalg.norm(selected_hf)) + 1e-12)
                )
            decision = {
                "pde": "grayscott",
                "task": "static_initial_condition_design",
                "model": model,
                "state_idx": state_idx,
                "state_hash": canonical_hashes[state_idx],
                "risk_limit": limit,
                "selected_idx": selected,
                "hf_oracle_idx": oracle,
                "surrogate_abstained": int(selected is None),
                "hf_feasible_set_empty": int(oracle is None),
                "selected_false_safe": int(
                    selected is not None and risk_sur[selected] <= limit and risk_hf[selected] > limit
                ),
                "selected_matches_hf_oracle": int(selected is not None and selected == oracle),
                "normalized_safe_regret": normalized_regret,
                "candidate_task_spearman": float(correlation) if np.isfinite(correlation) else None,
                "selected_field_relative_l2": selected_field_error,
            }
            append_jsonl(decision_path, decision)
            existing_decisions[(model, state_idx)] = decision
            print(
                f"[static design] model={model} state={state_idx} selected={selected} "
                f"oracle={oracle} false_safe={decision['selected_false_safe']}",
                flush=True,
            )

    decisions = list(iter_jsonl(decision_path))
    expected = len(models) * args.states
    if len(decisions) != expected:
        raise RuntimeError(f"Incomplete static-design decisions: {len(decisions)}/{expected}")
    finite_regret = [row["normalized_safe_regret"] for row in decisions if row["normalized_safe_regret"] is not None]
    payload = {
        "experiment": "Static PDE design",
        "role": "independent non-control decision-task confirmation",
        "models": models,
        "physical_state_groups": args.states,
        "model_state_decisions": len(decisions),
        "candidate_count_per_state": 16,
        "selected_false_safe": int(sum(row["selected_false_safe"] for row in decisions)),
        "selected_false_safe_fraction": float(np.mean([row["selected_false_safe"] for row in decisions])),
        "exact_hf_oracle_selection": int(sum(row["selected_matches_hf_oracle"] for row in decisions)),
        "exact_hf_oracle_selection_fraction": float(
            np.mean([row["selected_matches_hf_oracle"] for row in decisions])
        ),
        "mean_normalized_safe_regret": float(np.mean(finite_regret)),
        "median_candidate_task_spearman": float(
            np.median([row["candidate_task_spearman"] for row in decisions if row["candidate_task_spearman"] is not None])
        ),
        "median_selected_field_relative_l2": float(
            np.median([row["selected_field_relative_l2"] for row in decisions if row["selected_field_relative_l2"] is not None])
        ),
        "cluster_bootstrap_95ci": grouped_bootstrap(decisions, args.bootstrap, args.seed),
        "interpretation_guard": (
            "Rates apply to the predefined localized-design bank and deliberately selected "
            "high-gradient state cohort; they are not population prevalence estimates."
        ),
    }
    (args.outdir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

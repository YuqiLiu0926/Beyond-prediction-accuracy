#!/usr/bin/env python3
"""Generate controlled thermal trajectories for the held-out FNO surrogate."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from thermal_dynamics import (
    FINAL_CONFIG,
    build_grid,
    config_dict,
    random_block_controls,
    render_initial_state,
    rollout_controls,
    sample_ic_parameters,
    thermal_risk,
)


BASE = Path(__file__).resolve().parents[1]


def generate_one(task):
    index, parameters, control, grid_size, substeps = task
    grid = build_grid(FINAL_CONFIG, grid_size, substeps)
    initial = render_initial_state(parameters, grid)
    states = rollout_controls(initial, control, grid, store_states=True)
    return index, states.astype(np.float32), float(thermal_risk(states))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=BASE / "outputs/datasets/thermal_training_trajectories.npz",
    )
    parser.add_argument("--n-trajectories", type=int, default=240)
    parser.add_argument("--ic-seed", type=int, default=20260910)
    parser.add_argument("--control-seed", type=int, default=20260911)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    parameters = sample_ic_parameters(args.n_trajectories, args.ic_seed)
    controls = random_block_controls(args.n_trajectories, args.control_seed)
    tasks = [
        (index, parameters[index], controls[index], args.grid_size, args.substeps)
        for index in range(args.n_trajectories)
    ]
    states: list[np.ndarray | None] = [None] * args.n_trajectories
    risks = np.zeros(args.n_trajectories, dtype=np.float64)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate_one, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, trajectory, risk = future.result()
            states[index] = trajectory
            risks[index] = risk
            if completed % 20 == 0 or completed == args.n_trajectories:
                print(f"[thermal-data] {completed}/{args.n_trajectories}", flush=True)
    if any(value is None for value in states):
        raise RuntimeError("Missing generated trajectories")
    state_array = np.stack(states, axis=0)
    unsafe = risks > FINAL_CONFIG.T_ignition
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        state=state_array,
        control=controls.astype(np.float32),
        ic_parameters=np.asarray([json.dumps(row, sort_keys=True) for row in parameters]),
        risk=risks.astype(np.float32),
        config=np.asarray([json.dumps(config_dict(), sort_keys=True)]),
        grid_size=np.asarray([args.grid_size], dtype=np.int32),
        substeps=np.asarray([args.substeps], dtype=np.int32),
    )
    summary = {
        "status": "PASS",
        "output": str(args.out),
        "n_trajectories": args.n_trajectories,
        "state_shape": list(state_array.shape),
        "control_shape": list(controls.shape),
        "unsafe_count": int(np.sum(unsafe)),
        "unsafe_rate": float(np.mean(unsafe)),
        "risk_min": float(np.min(risks)),
        "risk_median": float(np.median(risks)),
        "risk_max": float(np.max(risks)),
        "ic_seed": args.ic_seed,
        "control_seed": args.control_seed,
        "grid_size": args.grid_size,
        "substeps": args.substeps,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

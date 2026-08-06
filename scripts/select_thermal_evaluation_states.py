#!/usr/bin/env python3
"""Select thermal challenge ICs without HF trajectory labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from run_thermal_closed_loop import (
    composite_family,
    load_model,
    physics_family_certificate,
    resize_block_average,
    surrogate_plan,
)
from thermal_dynamics import (
    FINAL_CONFIG,
    build_grid,
    continuation_only_controls,
    render_initial_state,
    sample_ic_parameters,
)
from train_thermal_surrogate import make_coords


BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ic-seed", type=int, required=True)
    parser.add_argument("--n-ics", type=int, default=100)
    parser.add_argument("--select", type=int, default=16)
    parser.add_argument("--planner-iterations", type=int, default=30)
    parser.add_argument("--apply-block", type=int, default=2)
    parser.add_argument("--control-block-size", type=int, default=2)
    parser.add_argument("--risk-limit", type=float, default=2.0)
    parser.add_argument("--deployment-beta", type=float, default=0.9)
    parser.add_argument("--coarse-allowance", type=float, default=0.0)
    parser.add_argument("--medium-allowance", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASE / "external_data/checkpoints/thermal/fno.pt",
    )
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, stats_t, checkpoint = load_model(args.checkpoint, device)
    coords = make_coords(32, device)
    deployment_config = replace(FINAL_CONFIG, beta=args.deployment_beta)
    grids = {
        "coarse": build_grid(deployment_config, 16, 5),
        "medium": build_grid(deployment_config, 32, 10),
        "hf_grid_only": build_grid(deployment_config, 64, 20),
    }
    parameters = sample_ic_parameters(args.n_ics, args.ic_seed, deployment_config)
    _, continuations = continuation_only_controls(deployment_config)
    rows = []
    for ic_idx, values in enumerate(parameters):
        # The 64-grid state is only an analytic rendering of the initial
        # condition. No HF time integration or outcome label is used here.
        state64 = render_initial_state(values, grids["hf_grid_only"], deployment_config)
        state32 = resize_block_average(state64, 32)
        candidate, plan = surrogate_plan(
            model,
            state32,
            stats_t,
            coords,
            device,
            args.planner_iterations,
            args.control_block_size,
            args.risk_limit,
        )
        _, current_certificate = physics_family_certificate(
            state64,
            [np.asarray(control) for control in continuations],
            grids["coarse"],
            grids["medium"],
            args.coarse_allowance,
            args.medium_allowance,
            args.risk_limit,
        )
        composites = composite_family(candidate, continuations, args.apply_block, FINAL_CONFIG.control_steps)
        _, candidate_certificate = physics_family_certificate(
            state64,
            composites,
            grids["coarse"],
            grids["medium"],
            args.coarse_allowance,
            args.medium_allowance,
            args.risk_limit,
        )
        rows.append(
            {
                "ic_idx": ic_idx,
                "initial_peak": float(np.max(state64[0])),
                "surrogate_risk": float(plan["surrogate_risk"]),
                "surrogate_feasible": int(plan["surrogate_feasible"]),
                "optimized_blend": float(plan["optimized_blend"]),
                "current_continuation_certified": int(current_certificate["certified"]),
                "current_continuation_level": current_certificate["level"],
                "current_continuation_upper": float(current_certificate["z_upper"]),
                "candidate_composite_certified": int(candidate_certificate["certified"]),
                "candidate_composite_level": candidate_certificate["level"],
                "candidate_composite_upper": float(candidate_certificate["z_upper"]),
            }
        )
        if (ic_idx + 1) % 10 == 0:
            print(f"[thermal-select] {ic_idx + 1}/{args.n_ics}", flush=True)

    eligible = [
        row
        for row in rows
        if row["surrogate_feasible"] and row["current_continuation_certified"]
    ]
    selected = sorted(
        eligible,
        key=lambda row: (
            -float(row["surrogate_risk"]),
            -float(row["candidate_composite_upper"]),
            int(row["ic_idx"]),
        ),
    )[: args.select]
    if len(selected) != args.select:
        raise RuntimeError(f"Only {len(selected)} eligible ICs; requested {args.select}")
    payload = {
        "status": "HF_TRAJECTORY_LABEL_FREE_SELECTION",
        "pde": "thermal_arrhenius",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "surrogate_training_beta": FINAL_CONFIG.beta,
        "deployment_beta": args.deployment_beta,
        "ic_seed": args.ic_seed,
        "n_ics": args.n_ics,
        "hf_time_integration_used": False,
        "selection_rule": (
            "Require surrogate-feasible candidate and current K5 physics certificate; "
            "rank by descending surrogate candidate risk, then descending composite physics upper bound."
        ),
        "eligible_count": len(eligible),
        "selected_indices": [int(row["ic_idx"]) for row in selected],
        "selected_rows": selected,
        "all_rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "eligible_count": len(eligible),
                "selected_indices": payload["selected_indices"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

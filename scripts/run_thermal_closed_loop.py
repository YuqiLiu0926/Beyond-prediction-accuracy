#!/usr/bin/env python3
"""Thermal closed-loop evaluation with verified-continuation preservation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from thermal_dynamics import (
    FINAL_CONFIG,
    build_grid,
    conversion_performance,
    continuation_only_controls,
    render_initial_state,
    rollout_controls_with_risk,
    sample_ic_parameters,
)
from train_thermal_surrogate import (
    ControlledThermalFNO,
    make_coords,
    rollout_model,
    tensor_stats,
)
from thermal_numerical_certificate import (
    ThermalCertificateCalibration,
    certify_control_family,
    resize_block_average,
)


BASE = Path(__file__).resolve().parents[1]
MODES = ("raw", "continuation_preserving", "continuation_only")


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ControlledThermalFNO(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats_np = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint["stats"].items()}
    return model, tensor_stats(stats_np, device), checkpoint


def surrogate_plan(
    model,
    state32: np.ndarray,
    stats_t,
    coords,
    device: torch.device,
    iterations: int,
    block_size: int,
    risk_limit: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    state_t = torch.from_numpy(np.asarray(state32, dtype=np.float32)[None]).to(device)
    n_blocks = int(np.ceil(FINAL_CONFIG.control_steps / block_size))
    raw = torch.full((n_blocks,), 0.25, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.08)
    best = None
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        block_values = 0.5 * torch.tanh(raw)
        controls = torch.repeat_interleave(block_values, block_size)[: FINAL_CONFIG.control_steps]
        states = rollout_model(model, state_t, controls[None], stats_t, coords)
        risk = torch.amax(states[:, :, 0])
        conversion = torch.mean(states[:, 0, 1]) - torch.mean(states[:, -1, 1])
        violation = torch.relu(risk - (risk_limit - 0.002))
        loss = -200.0 * conversion + 2500.0 * violation**2 + 0.002 * torch.mean(controls**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 5.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if best is None or value < best[0]:
            best = (value, controls.detach().cpu().numpy().astype(np.float64))
    if best is None:
        raise RuntimeError("Surrogate planner produced no iterate")

    optimized = best[1]
    cooling = np.full_like(optimized, FINAL_CONFIG.u_min)
    selected = cooling
    selected_risk = None
    selected_conversion = None
    selected_blend = 0.0
    selected_feasible = False
    least_risk_choice = None
    with torch.no_grad():
        for blend in np.linspace(1.0, 0.0, 17):
            candidate = blend * optimized + (1.0 - blend) * cooling
            controls_t = torch.from_numpy(candidate.astype(np.float32))[None].to(device)
            states_t = rollout_model(model, state_t, controls_t, stats_t, coords)
            risk = float(torch.amax(states_t[:, :, 0]).cpu())
            conversion = float(
                (torch.mean(states_t[:, 0, 1]) - torch.mean(states_t[:, -1, 1])).cpu()
            )
            if least_risk_choice is None or risk < least_risk_choice[0]:
                least_risk_choice = (risk, conversion, float(blend), candidate.copy())
            if risk <= risk_limit:
                selected = candidate
                selected_risk = risk
                selected_conversion = conversion
                selected_blend = float(blend)
                selected_feasible = True
                break
    if selected_risk is None:
        if least_risk_choice is None:
            raise RuntimeError("Surrogate feasibility line search produced no candidates")
        selected_risk, selected_conversion, selected_blend, selected = least_risk_choice
    return selected, {
        "surrogate_risk": selected_risk,
        "surrogate_conversion": selected_conversion,
        "optimized_blend": selected_blend,
        "surrogate_feasible": int(selected_feasible),
        "iterations": iterations,
    }


def physics_family_certificate(
    state64: np.ndarray,
    controls: list[np.ndarray],
    grids: dict[str, Any],
    calibration: ThermalCertificateCalibration,
    risk_limit: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    return certify_control_family(
        state64,
        controls,
        grids,
        calibration,
        risk_limit,
        rollout_controls_with_risk,
    )


def composite_family(
    candidate: np.ndarray, continuations: np.ndarray, apply_block: int, remaining: int
) -> list[np.ndarray]:
    return [
        np.concatenate([candidate[:apply_block], continuation[apply_block:remaining]])
        for continuation in continuations
    ]


def run_episode(
    model,
    stats_t,
    coords,
    initial64: np.ndarray,
    mode: str,
    continuations: np.ndarray,
    grids: dict[str, Any],
    calibration: ThermalCertificateCalibration,
    args,
) -> dict[str, Any]:
    state = initial64.copy()
    initial = initial64.copy()
    endpoint_states = [state.copy()]
    endpoint_risks = [float(np.max(state[0]))]
    peak_risk = endpoint_risks[0]
    controls_applied: list[float] = []
    decisions = []
    retained_continuation = None
    retained_certificate = None
    releases = 0
    interventions = 0
    unavailable = 0
    escalations = 0
    planning_seconds = 0.0
    certificate_seconds = 0.0

    for step in range(0, args.episode_steps, args.apply_block):
        block_len = min(args.apply_block, args.episode_steps - step)
        remaining = args.episode_steps - step
        remaining_continuations = np.asarray([value[:remaining] for value in continuations])
        current_continuation = None
        selected_continuation = None
        if mode in {"continuation_preserving", "continuation_only"} and (
            retained_continuation is None or mode == "continuation_only"
        ):
            selected_continuation, current_continuation = physics_family_certificate(
                state,
                [np.asarray(value) for value in remaining_continuations],
                grids,
                calibration,
                args.risk_limit,
            )
            certificate_seconds += current_continuation["N0_seconds"] + current_continuation["N1_seconds"]
            escalations += int(current_continuation["N1_evaluated"])
            if mode == "continuation_preserving" and int(current_continuation["certified"]):
                retained_continuation = selected_continuation.copy()
                retained_certificate = dict(current_continuation)

        candidate = None
        plan_info = None
        candidate_certificate = None
        selected_composite = None
        if mode != "continuation_only":
            started = time.perf_counter()
            state32 = resize_block_average(state, 32)
            candidate, plan_info = surrogate_plan(
                model,
                state32,
                stats_t,
                coords,
                args.device_t,
                args.planner_iterations,
                args.control_block_size,
                args.risk_limit,
            )
            planning_seconds += time.perf_counter() - started

        if mode == "raw":
            applied = candidate[:block_len]
            source = "surrogate_candidate"
            releases += 1
        elif mode == "continuation_only":
            if selected_continuation is None or current_continuation is None:
                raise RuntimeError("K5 continuation was not evaluated")
            applied = selected_continuation[:block_len]
            source = "continuation_only"
            unavailable += int(not current_continuation["certified"])
        else:
            composites = composite_family(candidate, remaining_continuations, args.apply_block, remaining)
            selected_composite, candidate_certificate = physics_family_certificate(
                state,
                composites,
                grids,
                calibration,
                args.risk_limit,
            )
            certificate_seconds += (
                candidate_certificate["N0_seconds"] + candidate_certificate["N1_seconds"]
            )
            escalations += int(candidate_certificate["N1_evaluated"])
            if int(candidate_certificate["certified"]):
                applied = candidate[:block_len]
                retained_continuation = selected_composite[args.apply_block :].copy()
                retained_certificate = dict(candidate_certificate)
                source = "surrogate_candidate"
                releases += 1
            elif retained_continuation is not None and len(retained_continuation) >= block_len:
                applied = retained_continuation[:block_len]
                retained_continuation = retained_continuation[block_len:].copy()
                source = "retained_continuation"
                interventions += 1
            else:
                if selected_continuation is None or current_continuation is None:
                    selected_continuation, current_continuation = physics_family_certificate(
                        state,
                        [np.asarray(value) for value in remaining_continuations],
                        grids,
                        calibration,
                        args.risk_limit,
                    )
                applied = selected_continuation[:block_len]
                source = "uncertified_emergency_continuation"
                interventions += 1
                unavailable += 1
        if source == "retained_continuation" and (
            retained_certificate is None or not retained_certificate["certified"]
        ):
            raise RuntimeError("Retained continuation lacks a valid certificate")

        states, block_peak = rollout_controls_with_risk(
            state, applied, grids["target"], store_states=True
        )
        state = states[-1]
        endpoint_states.extend(states[1:])
        endpoint_risks.extend(float(np.max(value[0])) for value in states[1:])
        peak_risk = max(peak_risk, block_peak)
        controls_applied.extend(float(value) for value in applied)
        decisions.append(
            {
                "step": step,
                "source": source,
                "current_risk": endpoint_risks[-block_len - 1],
                "current_continuation": current_continuation,
                "candidate_certificate": candidate_certificate,
                "plan_info": plan_info,
                "retained_length_after": (
                    len(retained_continuation) if retained_continuation is not None else 0
                ),
                "block_peak_hf_risk": block_peak,
            }
        )

    endpoint_array = np.stack(endpoint_states, axis=0)
    full_controls = np.asarray(controls_applied, dtype=np.float64)
    replay_states, replay_peak = rollout_controls_with_risk(
        initial, full_controls, grids["target"], store_states=True
    )
    replay_error = float(np.max(np.abs(replay_states - endpoint_array)))
    if replay_error > 1e-8 or abs(replay_peak - peak_risk) > 1e-8:
        raise RuntimeError((replay_error, replay_peak, peak_risk))
    return {
        "mode": mode,
        "unsafe": int(peak_risk > args.risk_limit),
        "peak_hf_risk": peak_risk,
        "risk_limit": args.risk_limit,
        "performance_conversion": conversion_performance(endpoint_array),
        "candidate_releases": releases,
        "interventions": interventions,
        "unavailable_decisions": unavailable,
        "N1_evaluations": escalations,
        "planning_seconds": planning_seconds,
        "certificate_seconds": certificate_seconds,
        "controls": controls_applied,
        "endpoint_risk_trace": endpoint_risks,
        "hf_replay_state_error": replay_error,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASE / "external_data/checkpoints/thermal/fno.pt",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=BASE / "outputs/closed_loop/thermal",
    )
    parser.add_argument("--ic-seed", type=int, default=20260920)
    parser.add_argument("--n-ics", type=int, default=8)
    parser.add_argument("--indices", default="")
    parser.add_argument("--modes", default="raw,continuation_preserving,continuation_only")
    parser.add_argument("--episode-steps", type=int, default=20)
    parser.add_argument("--apply-block", type=int, default=2)
    parser.add_argument("--control-block-size", type=int, default=2)
    parser.add_argument("--planner-iterations", type=int, default=60)
    parser.add_argument("--risk-limit", type=float, default=2.0)
    parser.add_argument("--deployment-beta", type=float, default=0.9)
    parser.add_argument(
        "--certificate-calibration",
        type=Path,
        default=BASE
        / "external_data/numerical_qualification/thermal/one_sided_allowance.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=int, default=1)
    args = parser.parse_args()
    args.device_t = torch.device(args.device if torch.cuda.is_available() else "cpu")
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    if any(mode not in MODES for mode in modes):
        raise ValueError(args.modes)
    if args.indices.strip():
        indices = [int(value) for value in args.indices.split(",")]
    else:
        indices = list(range(args.n_ics))
    if args.episode_steps != FINAL_CONFIG.control_steps:
        raise ValueError("The frozen held-out episode has exactly 20 control intervals")

    model, stats_t, checkpoint = load_model(args.checkpoint, args.device_t)
    calibration = ThermalCertificateCalibration.from_json(args.certificate_calibration)
    coords = make_coords(32, args.device_t)
    deployment_config = replace(FINAL_CONFIG, beta=args.deployment_beta)
    parameters = sample_ic_parameters(args.n_ics, args.ic_seed, deployment_config)
    grids = {
        "N0": build_grid(
            deployment_config, calibration.n0_grid_size, calibration.n0_substeps
        ),
        "N1": build_grid(
            deployment_config, calibration.n1_grid_size, calibration.n1_substeps
        ),
        "target": build_grid(
            deployment_config,
            calibration.target_grid_size,
            calibration.target_substeps,
        ),
    }
    hf_initials = [
        render_initial_state(value, grids["target"], deployment_config)
        for value in parameters
    ]
    names, continuations = continuation_only_controls(deployment_config)

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows_path = args.outdir / "episode_rows.jsonl"
    if not args.resume and rows_path.exists():
        rows_path.unlink()
    existing = []
    if rows_path.exists():
        existing = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    done = {(int(row["ic_idx"]), row["mode"]) for row in existing}
    for ic_idx in indices:
        for mode in modes:
            if (ic_idx, mode) in done:
                continue
            row = run_episode(
                model,
                stats_t,
                coords,
                hf_initials[ic_idx],
                mode,
                continuations,
                grids,
                calibration,
                args,
            )
            row.update(
                {
                    "pde": "thermal_arrhenius",
                    "model": "fno",
                    "ic_seed": args.ic_seed,
                    "ic_idx": ic_idx,
                    "continuation_names": names,
                }
            )
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"[thermal-loop] ic={ic_idx} mode={mode} unsafe={row['unsafe']} "
                f"risk={row['peak_hf_risk']:.4f} J={row['performance_conversion']:.6f} "
                f"release={row['candidate_releases']}",
                flush=True,
            )

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    expected = {(ic_idx, mode) for ic_idx in indices for mode in modes}
    selected = [row for row in rows if (int(row["ic_idx"]), row["mode"]) in expected]
    if len(selected) != len(expected) or {(int(row["ic_idx"]), row["mode"]) for row in selected} != expected:
        raise RuntimeError("Incomplete or duplicate thermal episodes")
    by_mode = {}
    for mode in modes:
        local = [row for row in selected if row["mode"] == mode]
        decisions = sum(len(row["decisions"]) for row in local)
        by_mode[mode] = {
            "episodes": len(local),
            "unsafe_episodes": sum(int(row["unsafe"]) for row in local),
            "mean_performance_conversion": float(
                np.mean([row["performance_conversion"] for row in local])
            ),
            "candidate_release_fraction": float(
                sum(int(row["candidate_releases"]) for row in local) / decisions
            ),
            "unavailable_decisions": sum(int(row["unavailable_decisions"]) for row in local),
            "interventions": sum(int(row["interventions"]) for row in local),
            "N1_evaluations": sum(int(row["N1_evaluations"]) for row in local),
            "mean_planning_seconds_per_decision": float(
                sum(float(row["planning_seconds"]) for row in local) / decisions
            ),
            "mean_certificate_seconds_per_decision": float(
                sum(float(row["certificate_seconds"]) for row in local) / decisions
            ),
        }
    summary = {
        "status": "DEVELOPMENT_ANALYSIS",
        "ic_seed": args.ic_seed,
        "indices": indices,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "risk_limit": args.risk_limit,
        "surrogate_training_beta": FINAL_CONFIG.beta,
        "deployment_beta": args.deployment_beta,
        "N0_allowance": calibration.n0_allowance,
        "N1_allowance": calibration.n1_allowance,
        "richardson_gamma": calibration.richardson_gamma,
        "grids": {
            "N0": f"{calibration.n0_grid_size}x{calibration.n0_grid_size}/{calibration.n0_substeps}",
            "N1": f"{calibration.n1_grid_size}x{calibration.n1_grid_size}/{calibration.n1_substeps}",
            "target": f"{calibration.target_grid_size}x{calibration.target_grid_size}/{calibration.target_substeps}",
        },
        "by_mode": by_mode,
        "rows": str(rows_path),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

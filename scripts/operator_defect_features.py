"""Compute nineteen PDE operator-defect features from surrogate rollouts.

The feature calculation uses the known discrete PDE operator, the predicted
state trajectory, and the proposed controls. High-fidelity trajectories are
not inputs. They are joined later only to fit and evaluate mismatch models.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from surrogate_pair_evaluator import Config, PairEvaluator


BASE = Path(__file__).resolve().parents[1]
PDES = ("burgers", "grayscott", "kolmogorov")
MODELS = ("deeponet", "fno", "pinn", "pino")

RESIDUAL_FEATURE_NAMES = (
    "pde_res_rel_mean",
    "pde_res_rel_q90",
    "pde_res_rel_max",
    "pde_res_abs_mean",
    "pde_res_abs_q95_mean",
    "pde_res_abs_max",
    "rhs_l2_mean",
    "state_dt_l2_mean",
    "conservation_res_mean",
    "conservation_res_max",
    "dissipation_res_mean",
    "dissipation_res_max",
    "periodic_jump_ratio_mean",
    "periodic_jump_ratio_max",
    "periodic_jump_abs_mean",
    "range_violation_mean",
    "range_violation_max",
    "residual_growth",
    "residual_peak_time",
)


def safe_mean(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else 0.0


def safe_max(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    return float(np.max(array)) if len(array) else 0.0


def safe_quantile(values: Any, quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    return float(np.quantile(array, quantile)) if len(array) else 0.0


def summarize_series(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"mean": 0.0, "q90": 0.0, "max": 0.0, "growth": 0.0, "peak_time": 0.0}
    peak = int(np.argmax(array))
    return {
        "mean": float(np.mean(array)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "growth": float(array[-1] - array[0]) if len(array) > 1 else 0.0,
        "peak_time": float(peak / max(1, len(array) - 1)),
    }


def diff_periodic_1d(values: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(values, -1, axis=-1) - np.roll(values, 1, axis=-1)) / (2.0 * dx)


def lap_periodic_1d(values: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(values, -1, axis=-1) - 2.0 * values + np.roll(values, 1, axis=-1)
    ) / (dx * dx)


def lap_periodic_2d(values: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return (
        (np.roll(values, -1, axis=-1) - 2.0 * values + np.roll(values, 1, axis=-1)) / (dx * dx)
        + (np.roll(values, -1, axis=-2) - 2.0 * values + np.roll(values, 1, axis=-2)) / (dy * dy)
    )


def periodic_jump_1d(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    if len(values) < 3:
        return 0.0, 0.0
    seam = float(abs(values[0] - values[-1]))
    interior = float(np.mean(np.abs(np.diff(values)))) + 1e-12
    return seam, seam / interior


def periodic_jump_2d(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        channel_values = [periodic_jump_2d(channel) for channel in values]
        return safe_mean([item[0] for item in channel_values]), safe_mean(
            [item[1] for item in channel_values]
        )
    if min(values.shape) < 3:
        return 0.0, 0.0
    seam_x = np.mean(np.abs(values[:, 0] - values[:, -1]))
    seam_y = np.mean(np.abs(values[0, :] - values[-1, :]))
    interior_x = np.mean(np.abs(np.diff(values, axis=1)))
    interior_y = np.mean(np.abs(np.diff(values, axis=0)))
    seam = float(0.5 * (seam_x + seam_y))
    interior = float(0.5 * (interior_x + interior_y)) + 1e-12
    return seam, seam / interior


def normalize_states(evaluator: PairEvaluator, states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim >= 1 and states.shape[0] == 1:
        states = states[0]
    if evaluator.pde == "burgers":
        if states.ndim == 3 and states.shape[1] == 1:
            states = states[:, 0, :]
        if states.ndim != 2:
            raise ValueError(f"Unexpected Burgers state shape: {states.shape}")
        return states
    if states.ndim == 5 and states.shape[0] == 1:
        states = states[0]
    if states.ndim != 4:
        raise ValueError(f"Unexpected {evaluator.pde} state shape: {states.shape}")
    return states


def selected_time_indices(horizon: int, maximum: int) -> list[int]:
    if horizon <= 0:
        return []
    maximum = max(1, int(maximum))
    if horizon <= maximum:
        return list(range(horizon))
    return sorted({int(index) for index in np.linspace(0, horizon - 1, maximum)})


def kolmogorov_rhs(evaluator: PairEvaluator, omega: np.ndarray, control: float) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.float64)
    ny, nx = omega.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=float(evaluator.dx))
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=float(evaluator.dy))
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k2 = kx_grid**2 + ky_grid**2
    inv_k2 = np.zeros_like(k2)
    inv_k2[k2 > 0.0] = 1.0 / k2[k2 > 0.0]
    omega_hat = np.fft.fft2(omega)
    stream_hat = omega_hat * inv_k2
    stream_hat[0, 0] = 0.0
    velocity_x = np.fft.ifft2(1j * ky_grid * stream_hat).real
    velocity_y = np.fft.ifft2(-1j * kx_grid * stream_hat).real
    omega_x = np.fft.ifft2(1j * kx_grid * omega_hat).real
    omega_y = np.fft.ifft2(1j * ky_grid * omega_hat).real
    laplacian = np.fft.ifft2(-k2 * omega_hat).real
    _, y_grid = np.meshgrid(evaluator.x, evaluator.y)
    forcing = float(evaluator.forcing_amp) * float(evaluator.forcing_k) * np.cos(
        float(evaluator.forcing_k) * y_grid
    )
    return (
        -(velocity_x * omega_x + velocity_y * omega_y)
        + float(evaluator.nu) * laplacian
        + forcing
        + float(control) * np.asarray(evaluator.actuator, dtype=np.float64)
    )


def operator_defect_features(
    evaluator: PairEvaluator,
    states: np.ndarray,
    controls: np.ndarray,
    maximum_time_samples: int = 33,
) -> dict[str, float]:
    """Return the nineteen operator-defect summaries used in the manuscript."""
    states = normalize_states(evaluator, states)
    controls = np.asarray(controls, dtype=np.float64).ravel()
    horizon = min(len(controls), len(states) - 1)
    indices = selected_time_indices(horizon, maximum_time_samples)
    if not indices:
        return {name: 0.0 for name in RESIDUAL_FEATURE_NAMES}

    relative, absolute_mean, absolute_q95, absolute_max = [], [], [], []
    rhs_norms, derivative_norms = [], []
    conservation, dissipation, jump_absolute, jump_ratio, range_violation = [], [], [], [], []
    dt = float(evaluator.dt_nom)

    for time_index in indices:
        control = float(controls[time_index])
        if evaluator.pde == "burgers":
            state = np.asarray(states[time_index], dtype=np.float64)
            next_state = np.asarray(states[time_index + 1], dtype=np.float64)
            state_dt = (next_state - state) / dt
            rhs = (
                -state * diff_periodic_1d(state, float(evaluator.dx_nom))
                + float(evaluator.nu) * lap_periodic_1d(state, float(evaluator.dx_nom))
                + control * evaluator.k_nom
            )
            defect = state_dt - rhs
            conservation_value = abs(float(np.mean(state_dt) - control * np.mean(evaluator.k_nom)))
            energy_dt = float((0.5 * np.mean(next_state**2) - 0.5 * np.mean(state**2)) / dt)
            energy_rhs = float(np.mean(state * rhs))
            dissipation_value = abs(energy_dt - energy_rhs)
            jump_abs, jump_rel = periodic_jump_1d(state)
            range_value = 0.0
        elif evaluator.pde == "grayscott":
            state = np.asarray(states[time_index], dtype=np.float64)
            next_state = np.asarray(states[time_index + 1], dtype=np.float64)
            u, v = state
            next_u, next_v = next_state
            du_dt = (next_u - u) / dt
            dv_dt = (next_v - v) / dt
            reaction = u * v**2
            rhs_u = float(evaluator.Du) * lap_periodic_2d(u, evaluator.dx, evaluator.dy) - reaction + float(evaluator.F0) * (1.0 - u)
            rhs_v = float(evaluator.Dv) * lap_periodic_2d(v, evaluator.dx, evaluator.dy) + reaction - (float(evaluator.F0) + float(evaluator.k0)) * v + control * evaluator.actuator
            state_dt = np.stack([du_dt, dv_dt])
            rhs = np.stack([rhs_u, rhs_v])
            defect = state_dt - rhs
            mass_u = float(np.mean(du_dt) - np.mean(-reaction + float(evaluator.F0) * (1.0 - u)))
            mass_v = float(np.mean(dv_dt) - np.mean(reaction - (float(evaluator.F0) + float(evaluator.k0)) * v + control * evaluator.actuator))
            conservation_value = math.sqrt(mass_u**2 + mass_v**2)
            energy_dt = float((0.5 * np.mean(next_u**2 + next_v**2) - 0.5 * np.mean(u**2 + v**2)) / dt)
            energy_rhs = float(np.mean(u * rhs_u + v * rhs_v))
            dissipation_value = abs(energy_dt - energy_rhs)
            jump_abs, jump_rel = periodic_jump_2d(state)
            lower = float(getattr(evaluator.model, "clip_min", 0.0))
            upper = float(getattr(evaluator.model, "clip_max", 1.5))
            range_value = float(np.mean(np.maximum(0.0, lower - state) + np.maximum(0.0, state - upper)))
        else:
            state = np.asarray(states[time_index, 0], dtype=np.float64)
            next_state = np.asarray(states[time_index + 1, 0], dtype=np.float64)
            state_dt = (next_state - state) / dt
            rhs = kolmogorov_rhs(evaluator, state, control)
            defect = state_dt - rhs
            forcing_mean = safe_mean(float(evaluator.forcing_amp) * float(evaluator.forcing_k) * np.cos(float(evaluator.forcing_k) * evaluator.Y))
            conservation_value = abs(float(np.mean(state_dt) - forcing_mean - control * np.mean(evaluator.actuator)))
            energy_dt = float((0.5 * np.mean(next_state**2) - 0.5 * np.mean(state**2)) / dt)
            energy_rhs = float(np.mean(state * rhs))
            dissipation_value = abs(energy_dt - energy_rhs)
            jump_abs, jump_rel = periodic_jump_2d(state)
            range_value = 0.0

        defect = np.asarray(defect, dtype=np.float64)
        rhs = np.asarray(rhs, dtype=np.float64)
        state_dt = np.asarray(state_dt, dtype=np.float64)
        absolute = np.abs(defect).ravel()
        rhs_norm = float(np.sqrt(np.mean(rhs**2)))
        derivative_norm = float(np.sqrt(np.mean(state_dt**2)))
        defect_norm = float(np.sqrt(np.mean(defect**2)))
        relative.append(defect_norm / (rhs_norm + derivative_norm + 1e-12))
        absolute_mean.append(float(np.mean(absolute)))
        absolute_q95.append(safe_quantile(absolute, 0.95))
        absolute_max.append(float(np.max(absolute)))
        rhs_norms.append(rhs_norm)
        derivative_norms.append(derivative_norm)
        conservation.append(conservation_value)
        dissipation.append(dissipation_value)
        jump_absolute.append(jump_abs)
        jump_ratio.append(jump_rel)
        range_violation.append(range_value)

    relative_summary = summarize_series(relative)
    return {
        "pde_res_rel_mean": relative_summary["mean"],
        "pde_res_rel_q90": relative_summary["q90"],
        "pde_res_rel_max": relative_summary["max"],
        "pde_res_abs_mean": safe_mean(absolute_mean),
        "pde_res_abs_q95_mean": safe_mean(absolute_q95),
        "pde_res_abs_max": safe_max(absolute_max),
        "rhs_l2_mean": safe_mean(rhs_norms),
        "state_dt_l2_mean": safe_mean(derivative_norms),
        "conservation_res_mean": safe_mean(conservation),
        "conservation_res_max": safe_max(conservation),
        "dissipation_res_mean": safe_mean(dissipation),
        "dissipation_res_max": safe_max(dissipation),
        "periodic_jump_ratio_mean": safe_mean(jump_ratio),
        "periodic_jump_ratio_max": safe_max(jump_ratio),
        "periodic_jump_abs_mean": safe_mean(jump_absolute),
        "range_violation_mean": safe_mean(range_violation),
        "range_violation_max": safe_max(range_violation),
        "residual_growth": relative_summary["growth"],
        "residual_peak_time": relative_summary["peak_time"],
    }


def surrogate_states(evaluator: PairEvaluator, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    if evaluator.pde == "burgers":
        return evaluator.mod.surrogate_rollout(
            evaluator.model, evaluator.model_type, initial_state, controls, evaluator.x_norm
        )
    return evaluator.mod.surrogate_rollout(
        evaluator.model, evaluator.model_type, initial_state, controls, evaluator.coords_2d
    )


def surrogate_risk(evaluator: PairEvaluator, states: np.ndarray) -> float:
    if evaluator.pde == "burgers":
        return float(evaluator.mod.risk_Z_inf(states, evaluator.dx_nom))
    if evaluator.pde == "grayscott":
        return float(evaluator.mod.risk_Z_front(states, evaluator.dx, evaluator.dy))
    return float(evaluator.mod.risk_Z_inf(states, evaluator.dx, evaluator.dy))


def surrogate_metrics_from_states(
    evaluator: PairEvaluator, states: np.ndarray
) -> dict[str, Any]:
    """Evaluate risk and task summaries from one surrogate trajectory."""
    if evaluator.pde == "burgers":
        risk_trace = evaluator.mod.risk_inf_series(states, evaluator.dx_nom)
        task_trace = evaluator.mod.response_series_1d(states, evaluator.k_nom)
        risk = evaluator.mod.risk_Z_inf(states, evaluator.dx_nom)
        task = evaluator.mod.terminal_response(states[-1], evaluator.k_nom)
    elif evaluator.pde == "grayscott":
        risk_trace = evaluator.mod.risk_front_series(states, evaluator.dx, evaluator.dy)
        task_trace = evaluator.mod.response_series(states, evaluator.actuator)
        risk = evaluator.mod.risk_Z_front(states, evaluator.dx, evaluator.dy)
        task = evaluator.mod.nominal_performance(states, evaluator.actuator)
    else:
        risk_trace = evaluator.mod.risk_inf_series(states, evaluator.dx, evaluator.dy)
        task_trace = evaluator.mod.response_series(states, evaluator.actuator)
        risk = evaluator.mod.risk_Z_inf(states, evaluator.dx, evaluator.dy)
        task = evaluator.mod.nominal_performance(states, evaluator.actuator)
    return {
        "z_sur_replay": float(risk),
        "J_sur_replay": float(task),
        "z_sur_ts_replay": np.asarray(risk_trace, dtype=np.float64),
        "J_sur_ts_replay": np.asarray(task_trace, dtype=np.float64),
    }


def configure_evaluator(
    data_root: Path,
    split: str,
    pde: str,
    model: str,
    archive_config: dict[str, Any],
    horizon: int,
    device: str,
) -> PairEvaluator:
    config = Config(
        root=str(data_root / "decision_archives" / split),
        ckpt_root=str(data_root / "checkpoints"),
        data_root=str(data_root / "datasets" / "discovery"),
        train_script=str(BASE / "scripts" / "train_discovery_surrogates.py"),
        burgers_script=str(BASE / "scripts" / "generate_burgers_decisions.py"),
        grayscott_script=str(BASE / "scripts" / "generate_gray_scott_decisions.py"),
        kolmogorov_script=str(BASE / "scripts" / "generate_kolmogorov_decisions.py"),
        H=horizon,
        canonical_horizon=horizon,
        device=device,
    )
    evaluator = PairEvaluator(pde, model, config)
    for name in ("seed", "ood_amp_scale", "ood_hf_amp", "candidate_pool_mult"):
        if name in archive_config and hasattr(evaluator.exp_cfg, name):
            setattr(evaluator.exp_cfg, name, archive_config[name])
    mode = "challenge_conditions" if split == "discovery" else "ood_amplified"
    evaluator.refresh_ics(int(archive_config["n_eval_ic"]), mode)
    return evaluator


def extract_archive(
    data_root: Path,
    split: str,
    pde: str,
    model: str,
    maximum_time_samples: int,
    device: str,
    limit: int,
) -> list[dict[str, Any]]:
    archive_path = (
        data_root
        / "decision_archives"
        / split
        / pde
        / model
        / "challenge_conditions"
        / "decision_data.npz"
    )
    with np.load(archive_path, allow_pickle=True) as archive:
        controls = np.asarray(archive["U_all"], dtype=np.float64)
        saved_risk = np.asarray(archive["z_sur"], dtype=np.float64)
        archive_config = json.loads(str(archive["config"].item()))
    if limit > 0:
        controls = controls[:limit]
        saved_risk = saved_risk[:limit]
    evaluator = configure_evaluator(
        data_root, split, pde, model, archive_config, controls.shape[1], device
    )
    rows = []
    for index, control in enumerate(controls):
        states = surrogate_states(evaluator, evaluator.x0s[index], control)
        values = operator_defect_features(
            evaluator, states, control, maximum_time_samples
        )
        rows.append(
            {
                "pde": pde,
                "model": model,
                "pair": f"{pde}:{model}",
                "ic_idx": index,
                "z_sur_replay": surrogate_risk(evaluator, states),
                "z_sur_archive": float(saved_risk[index]),
                "residual_feature_names": list(RESIDUAL_FEATURE_NAMES),
                "residual_features": [values[name] for name in RESIDUAL_FEATURE_NAMES],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE / "external_data")
    parser.add_argument("--split", choices=("discovery", "detector_development", "evaluation"), default="discovery")
    parser.add_argument("--pdes", default=",".join(PDES))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--maximum-time-samples", type=int, default=33)
    parser.add_argument("--limit-per-pair", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=BASE / "outputs" / "operator_defect_features" / "residual_features.jsonl")
    args = parser.parse_args()

    pdes = tuple(value.strip() for value in args.pdes.split(",") if value.strip())
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    rows: list[dict[str, Any]] = []
    for pde in pdes:
        for model in models:
            rows.extend(
                extract_archive(
                    args.data_root,
                    args.split,
                    pde,
                    model,
                    args.maximum_time_samples,
                    args.device,
                    args.limit_per_pair,
                )
            )
            print(f"{pde}/{model}: complete", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

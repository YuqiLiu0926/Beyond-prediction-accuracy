#!/usr/bin/env python3
"""Shared controlled thermal reaction-diffusion utilities for thermal study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from thermal_pde import (
    ThermalRunawayConfig,
    actuator_profile,
    make_grid,
    thermal_runaway_step_RK4,
)


@dataclass(frozen=True)
class HeldoutThermalConfig:
    Lx: float = 1.0
    Ly: float = 1.0
    alpha_T: float = 0.001
    alpha_C: float = 0.0005
    beta: float = 0.8
    gamma: float = 0.15
    Ea: float = 1.0
    R: float = 1.0
    T_ignition: float = 2.0
    T_ambient: float = 0.3
    u_min: float = -0.5
    u_max: float = 0.5
    ctrl_center_x: float = 0.5
    ctrl_center_y: float = 0.5
    ctrl_sigma: float = 0.25
    control_dt: float = 0.025
    control_steps: int = 20


FINAL_CONFIG = HeldoutThermalConfig()


def solver_config(config: HeldoutThermalConfig, grid_size: int, substeps: int) -> ThermalRunawayConfig:
    if grid_size < 8 or substeps < 1:
        raise ValueError((grid_size, substeps))
    return ThermalRunawayConfig(
        nx=grid_size,
        ny=grid_size,
        Lx=config.Lx,
        Ly=config.Ly,
        alpha_T=config.alpha_T,
        alpha_C=config.alpha_C,
        beta=config.beta,
        gamma=config.gamma,
        Ea=config.Ea,
        R=config.R,
        T_crit=2.5,
        T_ignition=config.T_ignition,
        T_ambient=config.T_ambient,
        dt_nominal=config.control_dt / substeps,
        n_steps=config.control_steps * substeps,
        horizon=config.control_steps,
        u_min=config.u_min,
        u_max=config.u_max,
        ctrl_center_x=config.ctrl_center_x,
        ctrl_center_y=config.ctrl_center_y,
        ctrl_sigma=config.ctrl_sigma,
        n_trajectories=1,
        risk_T_threshold=config.T_ignition,
    )


def build_grid(config: HeldoutThermalConfig, grid_size: int, substeps: int) -> dict[str, Any]:
    cfg = solver_config(config, grid_size, substeps)
    x, y, X, Y, dx, dy = make_grid(cfg)
    kx = 2.0 * np.pi * np.fft.fftfreq(cfg.nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(cfg.ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    return {
        "cfg": cfg,
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "K2": KX**2 + KY**2,
        "actuator": actuator_profile(X, Y, cfg),
        "substeps": substeps,
    }


def sample_ic_parameters(n: int, seed: int, config: HeldoutThermalConfig = FINAL_CONFIG) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rows.append(
            {
                "cx": float(rng.uniform(0.2, 0.8) * config.Lx),
                "cy": float(rng.uniform(0.2, 0.8) * config.Ly),
                "sigma_hot": float(rng.uniform(0.05, 0.20)),
                "amplitude": float(rng.uniform(0.3, 1.5) * config.T_ignition),
                "temperature_modes": rng.normal(0.0, 0.012, size=4).tolist(),
                "concentration_modes": rng.normal(0.0, 0.012, size=4).tolist(),
            }
        )
    return rows


def render_initial_state(
    parameters: dict[str, Any], grid: dict[str, Any], config: HeldoutThermalConfig = FINAL_CONFIG
) -> np.ndarray:
    X = grid["X"]
    Y = grid["Y"]
    dx = np.minimum(np.abs(X - parameters["cx"]), config.Lx - np.abs(X - parameters["cx"]))
    dy = np.minimum(np.abs(Y - parameters["cy"]), config.Ly - np.abs(Y - parameters["cy"]))
    T = config.T_ambient + parameters["amplitude"] * np.exp(
        -(dx**2 + dy**2) / (2.0 * parameters["sigma_hot"] ** 2)
    )
    tx = 2.0 * np.pi * X / config.Lx
    ty = 2.0 * np.pi * Y / config.Ly
    basis = [np.sin(tx), np.cos(ty), np.sin(tx + ty), np.cos(2.0 * tx - ty)]
    for coefficient, value in zip(parameters["temperature_modes"], basis):
        T = T + float(coefficient) * value
    T = np.clip(T, 0.05, 0.9 * config.T_ignition)

    C = np.ones_like(T)
    for coefficient, value in zip(parameters["concentration_modes"], basis):
        C = C + float(coefficient) * value
    C = np.clip(C, 0.0, 1.5)
    return np.stack([T, C], axis=0).astype(np.float64)


def rollout_controls(
    initial_state: np.ndarray,
    controls: np.ndarray,
    grid: dict[str, Any],
    store_states: bool = True,
) -> np.ndarray:
    states, _ = rollout_controls_with_risk(initial_state, controls, grid, store_states)
    return states


def rollout_controls_with_risk(
    initial_state: np.ndarray,
    controls: np.ndarray,
    grid: dict[str, Any],
    store_states: bool = True,
) -> tuple[np.ndarray, float]:
    controls = np.asarray(controls, dtype=np.float64).reshape(-1)
    T = np.asarray(initial_state[0], dtype=np.float64).copy()
    C = np.asarray(initial_state[1], dtype=np.float64).copy()
    if T.shape != grid["X"].shape or C.shape != T.shape:
        raise ValueError(f"State/grid mismatch: {initial_state.shape}, {grid['X'].shape}")
    states = [np.stack([T, C], axis=0)] if store_states else []
    peak_temperature = float(np.max(T))
    cfg = grid["cfg"]
    for control in controls:
        for _ in range(int(grid["substeps"])):
            T, C = thermal_runaway_step_RK4(
                T,
                C,
                float(control),
                grid["actuator"],
                cfg,
                None,
                None,
                grid["K2"],
                cfg.dt_nominal,
            )
            peak_temperature = max(peak_temperature, float(np.max(T)))
        if store_states:
            states.append(np.stack([T, C], axis=0))
    if store_states:
        result = np.stack(states, axis=0).astype(np.float64)
    else:
        result = np.stack([T, C], axis=0).astype(np.float64)
    return result, peak_temperature


def thermal_risk(states: np.ndarray) -> float:
    states = np.asarray(states)
    if states.ndim == 3:
        return float(np.max(states[0]))
    return float(np.max(states[:, 0]))


def conversion_performance(states: np.ndarray) -> float:
    states = np.asarray(states)
    if states.ndim != 4:
        raise ValueError(states.shape)
    return float(np.mean(states[0, 1]) - np.mean(states[-1, 1]))


def random_block_controls(
    n: int, seed: int, config: HeldoutThermalConfig = FINAL_CONFIG
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    controls = []
    for index in range(n):
        if index % 10 == 0:
            control = np.full(config.control_steps, config.u_min)
        elif index % 10 == 1:
            control = np.zeros(config.control_steps)
        elif index % 10 == 2:
            control = np.full(config.control_steps, config.u_max)
        else:
            block_count = int(rng.integers(3, 7))
            values = rng.uniform(config.u_min, config.u_max, size=block_count)
            edges = np.linspace(0, config.control_steps, block_count + 1, dtype=int)
            control = np.zeros(config.control_steps, dtype=np.float64)
            for j, value in enumerate(values):
                control[edges[j] : edges[j + 1]] = value
        controls.append(control)
    return np.asarray(controls, dtype=np.float64)


def continuation_only_controls(config: HeldoutThermalConfig = FINAL_CONFIG) -> tuple[list[str], np.ndarray]:
    values = [0.0, 0.5 * config.u_min, config.u_min, 0.5 * config.u_max, config.u_max]
    names = ["zero", "cool_half", "cool_full", "heat_half", "heat_full"]
    controls = np.asarray(
        [np.full(config.control_steps, value, dtype=np.float64) for value in values]
    )
    return names, controls


def config_dict(config: HeldoutThermalConfig = FINAL_CONFIG) -> dict[str, Any]:
    return asdict(config)

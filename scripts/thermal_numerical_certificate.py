#!/usr/bin/env python3
"""Safety-margin-adaptive numerical certificate for the thermal PDE."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


RiskRollout = Callable[[np.ndarray, np.ndarray, dict[str, Any], bool], tuple[Any, float]]


@dataclass(frozen=True)
class ThermalCertificateCalibration:
    """Frozen numerical levels and one-sided risk allowances."""

    n0_allowance: float
    n1_allowance: float
    richardson_gamma: float
    n0_grid_size: int = 32
    n0_substeps: int = 10
    n1_grid_size: int = 64
    n1_substeps: int = 10
    target_grid_size: int = 64
    target_substeps: int = 20

    @classmethod
    def from_json(cls, path: Path) -> "ThermalCertificateCalibration":
        payload = json.loads(path.read_text(encoding="utf-8"))
        levels = payload["numerical_levels"]
        calibration = cls(
            n0_allowance=float(payload["N0_one_sided_allowance"]),
            n1_allowance=float(payload["N1_one_sided_allowance"]),
            richardson_gamma=float(payload["richardson_gamma"]),
            n0_grid_size=int(levels["N0"]["spatial_grid_size"]),
            n0_substeps=int(levels["N0"]["substeps_per_control_interval"]),
            n1_grid_size=int(levels["N1"]["spatial_grid_size"]),
            n1_substeps=int(levels["N1"]["substeps_per_control_interval"]),
            target_grid_size=int(levels["target"]["spatial_grid_size"]),
            target_substeps=int(levels["target"]["substeps_per_control_interval"]),
        )
        if calibration.n0_allowance < 0 or calibration.n1_allowance < 0:
            raise ValueError("One-sided numerical allowances must be nonnegative")
        if calibration.richardson_gamma < 0:
            raise ValueError("The Richardson coefficient must be nonnegative")
        return calibration


def resize_block_average(state: np.ndarray, target: int) -> np.ndarray:
    """Transfer a square periodic field to a nested lower-resolution grid."""

    state = np.asarray(state, dtype=np.float64)
    source = state.shape[-1]
    if state.shape[-2] != source or source % target:
        raise ValueError(f"Cannot block-average state with shape {state.shape} to {target}")
    factor = source // target
    return state.reshape(state.shape[0], target, factor, target, factor).mean(axis=(2, 4))


def certify_control_family(
    state_target: np.ndarray,
    controls: list[np.ndarray],
    grids: dict[str, Any],
    calibration: ThermalCertificateCalibration,
    risk_limit: float,
    rollout: RiskRollout,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the lowest upper-risk sequence using the frozen two-level hierarchy."""

    if not controls:
        raise ValueError("At least one complete control sequence is required")

    state_n0 = resize_block_average(state_target, calibration.n0_grid_size)
    started = time.perf_counter()
    z_n0 = np.asarray(
        [rollout(state_n0, control, grids["N0"], False)[1] for control in controls],
        dtype=np.float64,
    )
    n0_seconds = time.perf_counter() - started
    upper_n0 = z_n0 + calibration.n0_allowance
    selected_n0 = int(np.argmin(upper_n0))

    if float(upper_n0[selected_n0]) <= float(risk_limit):
        return np.asarray(controls[selected_n0], dtype=np.float64), {
            "certified": 1,
            "level": "N0",
            "selected": selected_n0,
            "z_N0": float(z_n0[selected_n0]),
            "z_N1": None,
            "resolution_defect": None,
            "richardson_gamma": calibration.richardson_gamma,
            "N0_allowance": calibration.n0_allowance,
            "N1_allowance": calibration.n1_allowance,
            "z_upper": float(upper_n0[selected_n0]),
            "N0_best_upper": float(upper_n0[selected_n0]),
            "N0_seconds": n0_seconds,
            "N1_seconds": 0.0,
            "N1_evaluated": 0,
        }

    started = time.perf_counter()
    z_n1 = np.asarray(
        [rollout(state_target, control, grids["N1"], False)[1] for control in controls],
        dtype=np.float64,
    )
    n1_seconds = time.perf_counter() - started
    defect = np.abs(z_n1 - z_n0)
    upper_n1 = (
        z_n1
        + calibration.richardson_gamma * defect
        + calibration.n1_allowance
    )
    selected_n1 = int(np.argmin(upper_n1))
    certified = int(float(upper_n1[selected_n1]) <= float(risk_limit))
    return np.asarray(controls[selected_n1], dtype=np.float64), {
        "certified": certified,
        "level": "N1" if certified else "unavailable",
        "selected": selected_n1,
        "z_N0": float(z_n0[selected_n1]),
        "z_N1": float(z_n1[selected_n1]),
        "resolution_defect": float(defect[selected_n1]),
        "richardson_gamma": calibration.richardson_gamma,
        "N0_allowance": calibration.n0_allowance,
        "N1_allowance": calibration.n1_allowance,
        "z_upper": float(upper_n1[selected_n1]),
        "N0_best_upper": float(upper_n0[selected_n0]),
        "N0_seconds": n0_seconds,
        "N1_seconds": n1_seconds,
        "N1_evaluated": 1,
    }

"""Spectral solver for the controlled Arrhenius reaction-diffusion system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ThermalRunawayConfig:
    """Numerical and physical parameters for the dimensionless thermal PDE."""

    nx: int = 64
    ny: int = 64
    Lx: float = 1.0
    Ly: float = 1.0
    alpha_T: float = 0.001
    alpha_C: float = 0.0005
    beta: float = 0.80
    gamma: float = 0.15
    Ea: float = 1.0
    R: float = 1.0
    T_crit: float = 2.5
    T_ignition: float = 2.0
    T_ambient: float = 0.3
    dt_nominal: float = 0.001
    n_steps: int = 200
    horizon: int = 100
    u_min: float = -0.50
    u_max: float = 0.50
    ctrl_center_x: float = 0.5
    ctrl_center_y: float = 0.5
    ctrl_sigma: float = 0.15
    n_trajectories: int = 500
    risk_T_threshold: float = 1.7


def make_grid(
    config: ThermalRunawayConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Construct a uniform periodic grid."""
    x = np.linspace(0.0, config.Lx, config.nx, endpoint=False)
    y = np.linspace(0.0, config.Ly, config.ny, endpoint=False)
    dx = config.Lx / config.nx
    dy = config.Ly / config.ny
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y, dx, dy


def actuator_profile(
    X: np.ndarray,
    Y: np.ndarray,
    config: ThermalRunawayConfig,
) -> np.ndarray:
    """Return the unit-peak periodic Gaussian actuator profile."""
    dx = np.minimum(
        np.abs(X - config.ctrl_center_x),
        config.Lx - np.abs(X - config.ctrl_center_x),
    )
    dy = np.minimum(
        np.abs(Y - config.ctrl_center_y),
        config.Ly - np.abs(Y - config.ctrl_center_y),
    )
    profile = np.exp(-(dx**2 + dy**2) / (2.0 * config.ctrl_sigma**2))
    return profile / (np.max(np.abs(profile)) + 1e-12)


def arrhenius_rate(T: np.ndarray, config: ThermalRunawayConfig) -> np.ndarray:
    """Evaluate the Arrhenius factor with the documented temperature clamp."""
    temperature = np.clip(T, 0.05, 10.0)
    return np.exp(-config.Ea / (config.R * temperature))


def thermal_runaway_rhs(
    T: np.ndarray,
    C: np.ndarray,
    control: float,
    actuator: np.ndarray,
    config: ThermalRunawayConfig,
    kx: Optional[np.ndarray],
    ky: Optional[np.ndarray],
    K2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate spectral right-hand sides for temperature and concentration."""
    del kx, ky
    T_hat = np.fft.fft2(T)
    C_hat = np.fft.fft2(C)
    reaction = C * arrhenius_rate(T, config)
    temperature_source = config.beta * reaction + float(control) * actuator
    concentration_source = -config.gamma * reaction
    dT_hat = -config.alpha_T * K2 * T_hat + np.fft.fft2(temperature_source)
    dC_hat = -config.alpha_C * K2 * C_hat + np.fft.fft2(concentration_source)
    return dT_hat, dC_hat


def thermal_runaway_step_RK4(
    T: np.ndarray,
    C: np.ndarray,
    control: float,
    actuator: np.ndarray,
    config: ThermalRunawayConfig,
    kx: Optional[np.ndarray],
    ky: Optional[np.ndarray],
    K2: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the thermal PDE by one fourth-order Runge-Kutta step."""

    def rhs(T_local: np.ndarray, C_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return thermal_runaway_rhs(
            T_local, C_local, control, actuator, config, kx, ky, K2
        )

    dT1, dC1 = rhs(T, C)
    T1 = np.fft.ifft2(np.fft.fft2(T) + 0.5 * dt * dT1).real
    C1 = np.clip(
        np.fft.ifft2(np.fft.fft2(C) + 0.5 * dt * dC1).real,
        0.0,
        None,
    )

    dT2, dC2 = rhs(T1, C1)
    T2 = np.fft.ifft2(np.fft.fft2(T) + 0.5 * dt * dT2).real
    C2 = np.clip(
        np.fft.ifft2(np.fft.fft2(C) + 0.5 * dt * dC2).real,
        0.0,
        None,
    )

    dT3, dC3 = rhs(T2, C2)
    T3 = np.fft.ifft2(np.fft.fft2(T) + dt * dT3).real
    C3 = np.clip(
        np.fft.ifft2(np.fft.fft2(C) + dt * dC3).real,
        0.0,
        None,
    )

    dT4, dC4 = rhs(T3, C3)
    T_new = np.fft.ifft2(
        np.fft.fft2(T) + (dt / 6.0) * (dT1 + 2.0 * dT2 + 2.0 * dT3 + dT4)
    ).real
    C_new = np.clip(
        np.fft.ifft2(
            np.fft.fft2(C)
            + (dt / 6.0) * (dC1 + 2.0 * dC2 + 2.0 * dC3 + dC4)
        ).real,
        0.0,
        None,
    )
    return T_new.astype(np.float64), C_new.astype(np.float64)

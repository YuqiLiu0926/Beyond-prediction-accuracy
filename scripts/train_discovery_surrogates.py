# -*- coding: utf-8 -*-
"""Train the twelve controlled-PDE surrogates used in the discovery study.

The implementation supports Burgers, Gray--Scott and Kolmogorov data
generation; FNO, PINO, DeepONet and PINN training; rollout-aware checkpoint
selection; and the prediction metrics reported in Supplementary Tables 1--3.

The principal training features are:
  1) train-set state/control normalization saved into every checkpoint;
  2) residual learning in normalized variables: model predicts y_{t+1}-y_t;
  3) CNN/Fourier branch for DeepONet and latent-coordinate PINN instead of flatten-only vanilla MLP;
  4) optional gradient-risk loss for the risk carrier variable;
  5) optional short rollout loss to improve multi-step stability;
  6) dynamic balancing of physics loss against data loss;
  7) risk-specific validation metrics and rollout metrics saved for model-quality control;
  8) differentiable bounded-state prediction for Gray--Scott to make raw rollout physically admissible;
  9) long-rollout curriculum with risk/gradient rollout losses and sparse rollout batches;
 10) multi-horizon validation checkpoint selection aligned with downstream decision horizons;
 11) optional local CNN refinement head for DeepONet to preserve reaction--diffusion fronts;
 12) PDE- and architecture-specific presets fixed before the decision study;
 13) scale-normalized increment fitting and rollout-first checkpoint selection.

The file is standalone: it can generate the same controlled Burgers, Gray--Scott,
and Kolmogorov datasets, train FNO/PINO/DeepONet/PINN surrogates, and save
checkpoints and evaluation results in the layout documented in the public data package.
"""
from __future__ import annotations

import os
import gc
import math
import json
import argparse
import warnings
import copy
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.fft import rfft, irfft, rfft2, irfft2

from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# Global
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] device = {DEVICE}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

CONFIG: Dict[str, Any] = {
    "result_root": "./outputs/training/discovery",
    "seed": 42,
    "dpi": 400,

    # training
    "epochs": 260,
    "lr": 3e-4,
    "weight_decay": 1e-6,
    "grad_clip": 1.0,
    "amp": True,

    # staged losses
    "physics_warmup_epochs": 50,
    "rollout_warmup_epochs": 50,
    "rollout_loss_weight": 0.45,
    "rollout_loss_ramp_epochs": 90,
    "rollout_k_start": 4,
    "rollout_k_max": {"burgers": 12, "grayscott": 28, "kolmogorov": 12},
    "rollout_k_ramp_epochs": 210,
    "gradient_loss_weight": {
        "burgers": 0.05,
        "grayscott": 0.08,
        "kolmogorov": 0.10,
    },
    "physics_loss_weight": {
        # For high-fidelity surrogate training, physics terms are kept as a
        # weak regularizer. The Gray--Scott PINO value remains unchanged.
        "burgers": {"pinn": 5e-4, "pino": 7.5e-4},
        "grayscott": {"pinn": 1e-4, "pino": 1e-4},
        "kolmogorov": {"pinn": 1e-4, "pino": 7.5e-5},
    },
    "physics_dynamic_balance": True,
    "physics_balance_clip": 100.0,

    # stability regularization for autoregressive rollout
    "bound_loss_weight_lower": {"burgers": 0.0, "grayscott": 2.0e-2, "kolmogorov": 0.0},
    "bound_loss_weight_upper": {"burgers": 0.0, "grayscott": 3.0e-3, "kolmogorov": 0.0},
    "bound_warmup_epochs": 35,
    "residual_step_limit": True,
    "residual_limit_quantile": 0.995,
    "residual_limit_multiplier": 2.0,
    "residual_scale_max_pairs": 20000,
    "evaluate_projected_rollout": True,
    "projection_clip_eps": 1.0e-6,

    # split
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "test_ratio": 0.2,

    # batch
    "operator_batch_size_1d": 32,
    "operator_batch_size_2d": 8,
    "coord_batch_size_1d": 16,
    "coord_batch_size_2d": 4,

    # FNO sizes
    "fno_width_1d": 32,
    "fno_modes_1d": 16,
    "fno_layers_1d": 4,
    "fno_width_2d": 32,
    "fno_modes_2d": 16,
    "fno_layers_2d": 4,

    # latent-coordinate models
    "latent_dim_1d": 192,
    "latent_dim_2d": 256,
    "coord_hidden_1d": 160,
    "coord_depth_1d": 4,
    "coord_hidden_2d": 256,
    "coord_depth_2d": 4,
    "deeponet_hidden_1d": 192,
    "deeponet_hidden_2d": 256,
    "fourier_freqs": [1, 2, 4, 8, 16],

    # NC12Boost architecture knobs. These are activated only by helper
    # functions below for Burgers/Kolmogorov models and Gray--Scott PINN.
    # The already validated Gray--Scott DeepONet/PINO/FNO paths keep their
    # original architecture and weights.
    "nc12boost_enabled": True,
    "pinn_latent_dim_1d_boost": 256,
    "pinn_latent_dim_2d_boost": 384,
    "pinn_coord_hidden_1d_boost": 224,
    "pinn_coord_depth_1d_boost": 5,
    "pinn_coord_hidden_2d_boost": 384,
    "pinn_coord_depth_2d_boost": 5,
    "deeponet_latent_dim_1d_boost": 256,
    "deeponet_hidden_1d_boost": 256,
    "deeponet_latent_dim_2d_boost": 384,
    "deeponet_hidden_2d_boost": 384,
    "fno_width_1d_boost": 48,
    "fno_modes_1d_boost": 24,
    "fno_width_2d_boost": 48,
    "fno_modes_2d_boost": 20,
    "pinn_local_refine_1d": True,
    "pinn_local_refine_2d": True,
    "deeponet_local_refine_1d": True,
    "universal_local_refine_1d": True,
    "universal_local_refine_1d_models": ["pinn", "fno", "pino"],
    "universal_local_refine_2d_models_boost": ["pinn", "fno", "pino"],
    "local_refine_width_1d": 64,
    "local_refine_scale_max_1d": 0.30,
    "pinn_local_refine_width_2d": 64,
    "pinn_local_refine_scale_max_2d": 0.30,
    "deeponet_local_refine_width_1d": 64,
    "deeponet_local_refine_scale_max_1d": 0.30,

    # rollout validation / checkpoint selection
    "rollout_steps_1d": 80,
    "rollout_steps_2d": 100,
    "num_rollout_eval_traj": 50,
    "num_val_rollout_eval_traj": 16,
    "rollout_val_eval_period": 10,
    "selection_rollout_warmup_epochs": 70,
    "selection_val_guard_ratio": 1.35,
    "stable_select_weights": {"val": 0.12, "raw_rollout": 1.00, "risk": 0.35, "lower_sat": 0.35, "clip_frac": 0.12},
    "default_checkpoint_selection": "best_stable",

    # controlled Burgers
    "burgers": {
        "n_traj": 1000,
        "nx": 128,
        "nt": 120,
        "L": 2.0 * np.pi,
        "nu": 0.01,
        "dt": 5e-4,
        "u_min": -0.75,
        "u_max": 0.75,
        "ctrl_center": np.pi,
        "ctrl_sigma": 0.45,
        "ic_amp2_min": 0.2,
        "ic_amp2_max": 0.6,
        "ic_amp4_min": -0.3,
        "ic_amp4_max": 0.3,
    },

    # controlled Gray--Scott
    "grayscott": {
        "n_traj": 1000,
        "nx": 48,
        "ny": 48,
        "nt": 100,
        "Lx": 2.0 * np.pi,
        "Ly": 2.0 * np.pi,
        "Du": 2.0e-5,
        "Dv": 1.0e-5,
        "F": 0.035,
        "k": 0.065,
        "dt": 1.0,
        "u_min": -0.20,
        "u_max": 0.20,
        "ctrl_center_x": np.pi,
        "ctrl_center_y": np.pi,
        "ctrl_sigma": 0.6,
    },

    # controlled Kolmogorov
    "kolmogorov": {
        "n_traj": 1000,
        "nx": 48,
        "ny": 48,
        "nt": 100,
        "Lx": 2.0 * np.pi,
        "Ly": 2.0 * np.pi,
        "nu": 0.05,
        "dt": 0.01,
        "forcing_amp": 1.0,
        "forcing_k": 4,
        "u_min": -0.50,
        "u_max": 0.50,
        "ctrl_center_x": np.pi,
        "ctrl_center_y": np.pi,
        "ctrl_sigma": 0.7,
    },
    # NC long-rollout additions
    "grayscott_bounded_state": True,
    "soft_clamp_beta": 25.0,
    # Model-specific rollout semantics weights. Gray--Scott PINN/DeepONet usually
    # need stronger risk/gradient rollout constraints than FNO/PINO.
    "risk_rollout_weight": {
        "burgers": {"fno": 0.05, "pino": 0.06, "deeponet": 0.08, "pinn": 0.07, "default": 0.05},
        # Keep validated Gray--Scott non-PINN values unchanged; reduce PINN to
        # avoid over-optimizing only the risk carrier at the expense of field accuracy.
        "grayscott": {"fno": 0.22, "pino": 0.25, "pinn": 0.25, "deeponet": 0.35, "default": 0.30},
        "kolmogorov": {"fno": 0.16, "pino": 0.18, "deeponet": 0.18, "pinn": 0.18, "default": 0.16},
    },
    "grad_rollout_weight": {
        "burgers": {"fno": 0.05, "pino": 0.06, "deeponet": 0.08, "pinn": 0.07, "default": 0.05},
        # Keep validated Gray--Scott non-PINN values unchanged.
        "grayscott": {"fno": 0.08, "pino": 0.10, "pinn": 0.12, "deeponet": 0.15, "default": 0.12},
        "kolmogorov": {"fno": 0.09, "pino": 0.10, "deeponet": 0.10, "pinn": 0.10, "default": 0.09},
    },
    # Extra repair loss for model/PDE pairs that learned an identity-like
    # one-step map but accumulated large autoregressive drift. It is zero for
    # all already-good pairs by default.
    "increment_loss_weight": {
        "burgers": {"deeponet": 0.80, "pino": 0.80, "default": 0.0},
        "grayscott": {"pinn": 0.35, "default": 0.0},
        "kolmogorov": {"deeponet": 0.70, "fno": 0.70, "pinn": 0.30, "default": 0.0},
    },
    # Small positive scale only used by repair presets; the base value stays
    # zero so protected good runs keep their original behavior.
    "repair_local_refine_scale_init": 0.10,
    "repair_deeponet_local_refine_scale_init_1d": 0.10,
    "rollout_train_every": 4,
    "rollout_train_max_batch": 3,
    "scheduled_sampling_max": 0.55,
    "scheduled_sampling_ramp_epochs": 180,
    "input_noise_std_norm": 0.005,
    "input_noise_warmup_epochs": 60,
    "rollout_select_horizons": [40, 60, 80, 100],
    "rollout_select_weights_horizon": {40: 0.10, 60: 0.20, 80: 0.45, 100: 0.25},
    "multi_horizon_max_traj_per_eval": 16,
    "deeponet_local_refine": True,
    # Universal lightweight local residual refinement for Gray--Scott non-DeepONet
    # models. DeepONet has its own in-class refine head, so this targets FNO/PINO/PINN.
    "universal_local_refine_2d": True,
    "universal_local_refine_models_2d": ["fno", "pino", "pinn"],
    "local_refine_width_2d": 48,
    "local_refine_scale_init": 0.0,
    "local_refine_scale_max": 0.35,
    "num_workers": 2,
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# Utilities and grid helpers
# ============================================================
def split_traj_ids(n: int, train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    train_ids = perm[:n_train]
    val_ids = perm[n_train:n_train + n_val]
    test_ids = perm[n_train + n_val:]
    return train_ids, val_ids, test_ids


def make_periodic_1d_grid(nx: int, L: float):
    x = np.linspace(0.0, L, nx, endpoint=False, dtype=np.float64)
    dx = L / nx
    return x, dx


def make_periodic_2d_grid(nx: int, ny: int, Lx: float, Ly: float):
    x = np.linspace(0.0, Lx, nx, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, Ly, ny, endpoint=False, dtype=np.float64)
    dx = Lx / nx
    dy = Ly / ny
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y, dx, dy


def make_control_sequence(nt: int, u_min: float, u_max: float, rng: np.random.Generator):
    a = rng.uniform(u_min, u_max, size=nt)
    if nt >= 3:
        a_s = a.copy()
        a_s[1:-1] = 0.25 * a[:-2] + 0.50 * a[1:-1] + 0.25 * a[2:]
        a = a_s
    return a.astype(np.float64)


def gaussian_profile_1d(x: np.ndarray, center: float, sigma: float, L: float):
    d = np.minimum(np.abs(x - center), L - np.abs(x - center))
    g = np.exp(-(d**2) / (2.0 * sigma**2))
    return (g / (np.max(np.abs(g)) + 1e-12)).astype(np.float64)


def gaussian_profile_2d(X: np.ndarray, Y: np.ndarray, cx: float, cy: float, sigma: float, Lx: float, Ly: float):
    dx = np.minimum(np.abs(X - cx), Lx - np.abs(X - cx))
    dy = np.minimum(np.abs(Y - cy), Ly - np.abs(Y - cy))
    G = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    return (G / (np.max(np.abs(G)) + 1e-12)).astype(np.float64)


# ============================================================
# Dataset generation: Burgers
# ============================================================
def burgers_rhs(u: np.ndarray, a: float, kappa: np.ndarray, dx: float, nu: float):
    ux = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
    uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx ** 2)
    return -(u * ux) + nu * uxx + a * kappa


def burgers_step_ssprk3(u: np.ndarray, a: float, kappa: np.ndarray, dx: float, nu: float, dt: float):
    f1 = burgers_rhs(u, a, kappa, dx, nu)
    u1 = u + dt * f1
    f2 = burgers_rhs(u1, a, kappa, dx, nu)
    u2 = 0.75 * u + 0.25 * (u1 + dt * f2)
    f3 = burgers_rhs(u2, a, kappa, dx, nu)
    un = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * f3)
    return un.astype(np.float64)


def burgers_random_ic(x: np.ndarray, cfg: dict, rng: np.random.Generator):
    amp2 = rng.uniform(cfg["ic_amp2_min"], cfg["ic_amp2_max"])
    amp4 = rng.uniform(cfg["ic_amp4_min"], cfg["ic_amp4_max"])
    p2 = rng.uniform(0.0, 2.0 * np.pi)
    p4 = rng.uniform(0.0, 2.0 * np.pi)
    u0 = np.sin(x) + amp2 * np.sin(2.0 * x + p2) + amp4 * np.sin(4.0 * x + p4)
    return u0.astype(np.float64)


def generate_burgers_controlled_dataset(cfg_all: dict):
    cfg = cfg_all["burgers"]
    x, dx = make_periodic_1d_grid(cfg["nx"], cfg["L"])
    kappa = gaussian_profile_1d(x, cfg["ctrl_center"], cfg["ctrl_sigma"], cfg["L"])
    rng = np.random.default_rng(cfg_all["seed"])
    U = np.zeros((cfg["n_traj"], cfg["nt"] + 1, cfg["nx"]), dtype=np.float32)
    A = np.zeros((cfg["n_traj"], cfg["nt"]), dtype=np.float32)
    for j in tqdm(range(cfg["n_traj"]), desc="Generate controlled Burgers"):
        u = burgers_random_ic(x, cfg, rng)
        a_seq = make_control_sequence(cfg["nt"], cfg["u_min"], cfg["u_max"], rng)
        U[j, 0] = u.astype(np.float32)
        A[j] = a_seq.astype(np.float32)
        for t in range(cfg["nt"]):
            u = burgers_step_ssprk3(u, float(a_seq[t]), kappa, dx, cfg["nu"], cfg["dt"])
            U[j, t + 1] = u.astype(np.float32)
    return {"state": U, "control": A, "grid_x": x.astype(np.float32), "actuator": kappa.astype(np.float32), "meta": cfg.copy()}


# ============================================================
# Dataset generation: Gray--Scott
# ============================================================
def laplacian_2d(z: np.ndarray, dx: float, dy: float):
    return (
        (np.roll(z, -1, axis=1) - 2.0 * z + np.roll(z, 1, axis=1)) / (dx ** 2)
        + (np.roll(z, -1, axis=0) - 2.0 * z + np.roll(z, 1, axis=0)) / (dy ** 2)
    )


def grayscott_step(u: np.ndarray, v: np.ndarray, a: float, gxy: np.ndarray,
                   dx: float, dy: float, Du: float, Dv: float, F0: float, k0: float, dt: float):
    Lu = laplacian_2d(u, dx, dy)
    Lv = laplacian_2d(v, dx, dy)
    uvv = u * (v ** 2)
    du = Du * Lu - uvv + F0 * (1.0 - u)
    dv = Dv * Lv + uvv - (F0 + k0) * v + a * gxy
    un = np.clip(u + dt * du, 0.0, 1.5)
    vn = np.clip(v + dt * dv, 0.0, 1.5)
    return un.astype(np.float64), vn.astype(np.float64)


def grayscott_random_ic(nx: int, ny: int, rng: np.random.Generator):
    u = np.ones((ny, nx), dtype=np.float64)
    v = np.zeros((ny, nx), dtype=np.float64)
    cx = rng.integers(nx // 3, 2 * nx // 3)
    cy = rng.integers(ny // 3, 2 * ny // 3)
    rad = rng.integers(max(3, nx // 12), max(5, nx // 8))
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
    u[mask] = 0.50 + 0.05 * rng.standard_normal(np.sum(mask))
    v[mask] = 0.25 + 0.05 * rng.standard_normal(np.sum(mask))
    u += 0.01 * rng.standard_normal((ny, nx))
    v += 0.01 * rng.standard_normal((ny, nx))
    return np.clip(u, 0.0, 1.5), np.clip(v, 0.0, 1.5)


def generate_grayscott_controlled_dataset(cfg_all: dict):
    cfg = cfg_all["grayscott"]
    x, y, X, Y, dx, dy = make_periodic_2d_grid(cfg["nx"], cfg["ny"], cfg["Lx"], cfg["Ly"])
    gxy = gaussian_profile_2d(X, Y, cfg["ctrl_center_x"], cfg["ctrl_center_y"], cfg["ctrl_sigma"], cfg["Lx"], cfg["Ly"])
    rng = np.random.default_rng(cfg_all["seed"])
    S = np.zeros((cfg["n_traj"], cfg["nt"] + 1, 2, cfg["ny"], cfg["nx"]), dtype=np.float32)
    A = np.zeros((cfg["n_traj"], cfg["nt"]), dtype=np.float32)
    for j in tqdm(range(cfg["n_traj"]), desc="Generate controlled GrayScott"):
        u, v = grayscott_random_ic(cfg["nx"], cfg["ny"], rng)
        a_seq = make_control_sequence(cfg["nt"], cfg["u_min"], cfg["u_max"], rng)
        S[j, 0, 0] = u.astype(np.float32)
        S[j, 0, 1] = v.astype(np.float32)
        A[j] = a_seq.astype(np.float32)
        for t in range(cfg["nt"]):
            u, v = grayscott_step(u, v, float(a_seq[t]), gxy, dx, dy, cfg["Du"], cfg["Dv"], cfg["F"], cfg["k"], cfg["dt"])
            S[j, t + 1, 0] = u.astype(np.float32)
            S[j, t + 1, 1] = v.astype(np.float32)
    return {"state": S, "control": A, "grid_x": x.astype(np.float32), "grid_y": y.astype(np.float32), "actuator": gxy.astype(np.float32), "meta": cfg.copy()}


# ============================================================
# Dataset generation: Kolmogorov
# ============================================================
def kolmogorov_random_ic(X: np.ndarray, Y: np.ndarray, k_forcing: int, rng: np.random.Generator):
    omega0 = k_forcing * np.cos(k_forcing * Y)
    pert = np.zeros_like(omega0)
    for mx in range(1, 5):
        for my in range(1, 5):
            a = rng.normal()
            b = rng.normal()
            pert += a * np.sin(mx * X + my * Y) + b * np.cos(mx * X - my * Y)
    pert = pert / (np.std(pert) + 1e-12)
    omega0 = omega0 + 0.20 * pert
    return omega0.astype(np.float64)


def kolmogorov_step(omega: np.ndarray, a: float, gamma: np.ndarray,
                    nu: float, dt: float, forcing_amp: float, forcing_k: int,
                    x: np.ndarray, y: np.ndarray):
    ny, nx = omega.shape
    Lx = x[-1] - x[0] + (x[1] - x[0])
    Ly = y[-1] - y[0] + (y[1] - y[0])
    dx = Lx / nx
    dy = Ly / ny
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    invK2 = np.zeros_like(K2)
    invK2[K2 > 0] = 1.0 / K2[K2 > 0]
    base_forcing = forcing_amp * forcing_k * np.cos(forcing_k * np.meshgrid(x, y)[1])

    def rhs(w):
        w_hat = np.fft.fft2(w)
        psi_hat = w_hat * invK2
        psi_hat[0, 0] = 0.0
        u = np.fft.ifft2(1j * KY * psi_hat).real
        v = np.fft.ifft2(-1j * KX * psi_hat).real
        wx = np.fft.ifft2(1j * KX * w_hat).real
        wy = np.fft.ifft2(1j * KY * w_hat).real
        lap = np.fft.ifft2(-K2 * w_hat).real
        return -(u * wx + v * wy) + nu * lap + base_forcing + a * gamma

    k1 = rhs(omega)
    k2 = rhs(omega + 0.5 * dt * k1)
    k3 = rhs(omega + 0.5 * dt * k2)
    k4 = rhs(omega + dt * k3)
    return (omega + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)).astype(np.float64)


def generate_kolmogorov_controlled_dataset(cfg_all: dict):
    cfg = cfg_all["kolmogorov"]
    x, y, X, Y, dx, dy = make_periodic_2d_grid(cfg["nx"], cfg["ny"], cfg["Lx"], cfg["Ly"])
    gamma = gaussian_profile_2d(X, Y, cfg["ctrl_center_x"], cfg["ctrl_center_y"], cfg["ctrl_sigma"], cfg["Lx"], cfg["Ly"])
    rng = np.random.default_rng(cfg_all["seed"])
    S = np.zeros((cfg["n_traj"], cfg["nt"] + 1, 1, cfg["ny"], cfg["nx"]), dtype=np.float32)
    A = np.zeros((cfg["n_traj"], cfg["nt"]), dtype=np.float32)
    for j in tqdm(range(cfg["n_traj"]), desc="Generate controlled Kolmogorov"):
        omega = kolmogorov_random_ic(X, Y, cfg["forcing_k"], rng)
        a_seq = make_control_sequence(cfg["nt"], cfg["u_min"], cfg["u_max"], rng)
        S[j, 0, 0] = omega.astype(np.float32)
        A[j] = a_seq.astype(np.float32)
        for t in range(cfg["nt"]):
            omega = kolmogorov_step(omega, float(a_seq[t]), gamma, cfg["nu"], cfg["dt"], cfg["forcing_amp"], cfg["forcing_k"], x, y)
            S[j, t + 1, 0] = omega.astype(np.float32)
    return {"state": S, "control": A, "grid_x": x.astype(np.float32), "grid_y": y.astype(np.float32), "actuator": gamma.astype(np.float32), "meta": cfg.copy()}


# ============================================================
# Normalization
# ============================================================
@dataclass
class NormalizerStats:
    state_mean: List[float]
    state_std: List[float]
    control_mean: float
    control_std: float
    residual_learning: bool = True


class Normalizer:
    def __init__(self, stats: NormalizerStats):
        self.stats = stats
        self.state_mean_np = np.asarray(stats.state_mean, dtype=np.float32)
        self.state_std_np = np.asarray(stats.state_std, dtype=np.float32)
        self.control_mean = float(stats.control_mean)
        self.control_std = float(stats.control_std)

    @staticmethod
    def fit(state: np.ndarray, control: np.ndarray, train_ids: np.ndarray) -> "Normalizer":
        st = state[train_ids]
        ctrl = control[train_ids]
        if st.ndim == 3:  # [J,T+1,N]
            st_c = st[:, :, None, :]
            mean = st_c.mean(axis=(0, 1, 3))
            std = st_c.std(axis=(0, 1, 3))
        elif st.ndim == 5:  # [J,T+1,C,H,W]
            mean = st.mean(axis=(0, 1, 3, 4))
            std = st.std(axis=(0, 1, 3, 4))
        else:
            raise ValueError(f"Unexpected state shape: {state.shape}")
        std = np.maximum(std, 1e-6)
        cm = float(ctrl.mean())
        cs = float(max(ctrl.std(), 1e-6))
        return Normalizer(NormalizerStats(mean.astype(float).tolist(), std.astype(float).tolist(), cm, cs, True))

    def state_to_channel(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x[None, :]
        return x

    def normalize_state_np(self, x: np.ndarray) -> np.ndarray:
        x = self.state_to_channel(x).astype(np.float32)
        if x.ndim == 2:  # [C,N]
            return (x - self.state_mean_np[:, None]) / self.state_std_np[:, None]
        if x.ndim == 3:  # [C,H,W]
            return (x - self.state_mean_np[:, None, None]) / self.state_std_np[:, None, None]
        if x.ndim == 4:  # [B,C,H,W] or [B,C,N] ambiguous impossible for 1d batch? use torch for batches
            return (x - self.state_mean_np[None, :, None, None]) / self.state_std_np[None, :, None, None]
        raise ValueError(f"normalize_state_np unexpected {x.shape}")

    def normalize_control_np(self, a: np.ndarray) -> np.ndarray:
        return ((a.astype(np.float32) - self.control_mean) / self.control_std).astype(np.float32)

    def state_mean_t(self, device=DEVICE, ndim: int = 3):
        t = torch.tensor(self.state_mean_np, dtype=torch.float32, device=device)
        if ndim == 3:  # [B,C,N]
            return t[None, :, None]
        if ndim == 4:  # [B,C,H,W]
            return t[None, :, None, None]
        raise ValueError(ndim)

    def state_std_t(self, device=DEVICE, ndim: int = 3):
        t = torch.tensor(self.state_std_np, dtype=torch.float32, device=device)
        if ndim == 3:
            return t[None, :, None]
        if ndim == 4:
            return t[None, :, None, None]
        raise ValueError(ndim)

    def denorm_state_t(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 3:
            return y * self.state_std_t(y.device, 3) + self.state_mean_t(y.device, 3)
        if y.ndim == 4:
            return y * self.state_std_t(y.device, 4) + self.state_mean_t(y.device, 4)
        raise ValueError(f"denorm_state_t unexpected {y.shape}")

    def norm_state_t(self, x_phys: torch.Tensor) -> torch.Tensor:
        if x_phys.ndim == 3:
            return (x_phys - self.state_mean_t(x_phys.device, 3)) / self.state_std_t(x_phys.device, 3)
        if x_phys.ndim == 4:
            return (x_phys - self.state_mean_t(x_phys.device, 4)) / self.state_std_t(x_phys.device, 4)
        raise ValueError(f"norm_state_t unexpected {x_phys.shape}")

    def norm_control_t(self, a_phys: torch.Tensor) -> torch.Tensor:
        return (a_phys - self.control_mean) / self.control_std

    def denorm_control_t(self, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm * self.control_std + self.control_mean


# ============================================================
# Normalized pair dataset
# ============================================================
class NormalizedPairDataset(Dataset):
    def __init__(self, state: np.ndarray, control: np.ndarray, traj_ids: np.ndarray, normalizer: Normalizer, pde_name: str):
        super().__init__()
        self.state = state
        self.control = control
        self.traj_ids = np.asarray(traj_ids, dtype=int)
        self.normalizer = normalizer
        self.pde_name = pde_name
        self.T = control.shape[1]
        self.is_1d = (state.ndim == 3)
        if self.is_1d:
            self.C = 1
            self.N = state.shape[-1]
        else:
            self.C = state.shape[2]
            self.H = state.shape[-2]
            self.W = state.shape[-1]
        self.index = [(int(tr), int(t)) for tr in self.traj_ids for t in range(self.T)]

    def __len__(self):
        return len(self.index)

    def get_state_norm_np(self, tr: int, t: int) -> np.ndarray:
        s = self.state[tr, t]
        return self.normalizer.normalize_state_np(s)

    def get_control_norm_np(self, tr: int, t: int) -> np.ndarray:
        a = np.array([self.control[tr, t]], dtype=np.float32)
        return self.normalizer.normalize_control_np(a)

    def __getitem__(self, idx):
        tr, t = self.index[idx]
        s0_n = self.get_state_norm_np(tr, t)
        s1_n = self.get_state_norm_np(tr, t + 1)
        delta_n = s1_n - s0_n
        a_phys = np.array([self.control[tr, t]], dtype=np.float32)
        a_norm = self.normalizer.normalize_control_np(a_phys)
        return (
            torch.from_numpy(s0_n.astype(np.float32)),
            torch.from_numpy(delta_n.astype(np.float32)),
            torch.from_numpy(s1_n.astype(np.float32)),
            torch.from_numpy(a_norm.astype(np.float32)),
            torch.from_numpy(a_phys.astype(np.float32)),
            torch.tensor(tr, dtype=torch.long),
            torch.tensor(t, dtype=torch.long),
        )



# ============================================================
# Stability helpers: residual step limiter and soft physical bounds
# ============================================================
def compute_residual_scale_np(state: np.ndarray, train_ids: np.ndarray, normalizer: Normalizer,
                              max_pairs: int = 20000, quantile: float = 0.995,
                              multiplier: float = 2.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = state.shape[1] - 1
    pairs = [(int(tr), int(t)) for tr in train_ids for t in range(T)]
    if len(pairs) > max_pairs:
        sel = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in sel]
    C = 1 if state.ndim == 3 else state.shape[2]
    vals_by_c = [[] for _ in range(C)]
    for tr, t in pairs:
        y0 = normalizer.normalize_state_np(state[tr, t])
        y1 = normalizer.normalize_state_np(state[tr, t + 1])
        d = np.abs(y1 - y0)
        for c in range(C):
            vals_by_c[c].append(d[c].reshape(-1))
    scales = []
    for c in range(C):
        vals = np.concatenate(vals_by_c[c]) if vals_by_c[c] else np.array([1.0], dtype=np.float32)
        q = float(np.quantile(vals, quantile))
        scales.append(max(multiplier * q, 1.0e-4))
    return np.asarray(scales, dtype=np.float32)


def residual_scale_t(residual_scale_np: Optional[np.ndarray], device: torch.device, ndim: int) -> Optional[torch.Tensor]:
    if residual_scale_np is None:
        return None
    t = torch.as_tensor(residual_scale_np, dtype=torch.float32, device=device)
    if ndim == 3:
        return t[None, :, None]
    if ndim == 4:
        return t[None, :, None, None]
    raise ValueError(ndim)


def apply_residual_step_limit(delta_raw: torch.Tensor, residual_scale_np: Optional[np.ndarray]) -> torch.Tensor:
    if (not CONFIG.get("residual_step_limit", True)) or residual_scale_np is None:
        return delta_raw
    scale = residual_scale_t(residual_scale_np, delta_raw.device, delta_raw.ndim)
    return scale * torch.tanh(delta_raw / (scale + 1.0e-12))



def get_pde_model_weight(key: str, pde_name: str, model_type: str, default: float = 0.0) -> float:
    """Read scalar, per-PDE, or per-PDE/per-model weights from CONFIG."""
    cfg = CONFIG.get(key, default)
    if isinstance(cfg, (int, float)):
        return float(cfg)
    if isinstance(cfg, dict):
        val = cfg.get(pde_name, cfg.get("default", default))
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            return float(val.get(model_type, val.get("default", default)))
    return float(default)

def is_protected_grayscott_model(pde_name: str, model_type: str) -> bool:
    """Models whose validated Gray--Scott paths should remain unchanged."""
    return pde_name == "grayscott" and model_type in {"deeponet", "pino", "fno"}


GOOD_PAIRS = {
    "burgers:fno",
    "burgers:pinn",
    "grayscott:deeponet",
    "grayscott:fno",
    "grayscott:pino",
    "kolmogorov:pino",
}

REPAIR_PAIRS = {
    "burgers:deeponet",
    "burgers:pino",
    "grayscott:pinn",
    "kolmogorov:deeponet",
    "kolmogorov:fno",
    "kolmogorov:pinn",
}

def is_good_pair(pde_name: str, model_type: str) -> bool:
    return f"{pde_name}:{model_type}" in GOOD_PAIRS

def is_repair_pair(pde_name: str, model_type: str) -> bool:
    return f"{pde_name}:{model_type}" in REPAIR_PAIRS


def is_nc12boost_active(pde_name: str, model_type: str) -> bool:
    """Activate boost for Burgers/Kolmogorov all models and Gray--Scott PINN only."""
    if not bool(CONFIG.get("nc12boost_enabled", True)):
        return False
    if is_protected_grayscott_model(pde_name, model_type):
        return False
    return pde_name in {"burgers", "kolmogorov"} or (pde_name == "grayscott" and model_type == "pinn")


def cfg_dim(name: str, pde_name: str, model_type: str) -> int:
    """Return boosted architecture dimensions only for intended NC12 cases."""
    if is_nc12boost_active(pde_name, model_type):
        boost_key = f"{name}_boost"
        if boost_key in CONFIG:
            return int(CONFIG[boost_key])
    return int(CONFIG[name])


def apply_model_training_preset(pde_name: str, model_type: str):
    """Apply per PDE/model training presets before dataset/model construction.

    This repair script freezes the six already-good pairs and applies stronger
    settings only to the six under-performing pairs:
      Burgers-DeepONet, Burgers-PINO, GrayScott-PINN,
      Kolmogorov-DeepONet, Kolmogorov-FNO, Kolmogorov-PINN.
    Command-line overrides are re-applied in main() after this preset.
    """
    if is_good_pair(pde_name, model_type) or is_protected_grayscott_model(pde_name, model_type):
        return

    # Keep the generic NC12Boost behavior for non-listed pairs, but the intended
    # usage of this file is to run only REPAIR_PAIRS.
    if not is_repair_pair(pde_name, model_type):
        return

    CONFIG["default_checkpoint_selection"] = "best_rollout"
    CONFIG["rollout_loss_weight"] = max(float(CONFIG.get("rollout_loss_weight", 0.45)), 0.85)
    CONFIG["scheduled_sampling_max"] = max(float(CONFIG.get("scheduled_sampling_max", 0.55)), 0.70)
    CONFIG["selection_val_guard_ratio"] = max(float(CONFIG.get("selection_val_guard_ratio", 1.35)), 3.0)
    CONFIG["rollout_train_every"] = min(int(CONFIG.get("rollout_train_every", 4)), 2)
    CONFIG["local_refine_scale_init"] = float(CONFIG.get("repair_local_refine_scale_init", 0.10))
    CONFIG["deeponet_local_refine_scale_init_1d"] = float(CONFIG.get("repair_deeponet_local_refine_scale_init_1d", 0.10))

    if pde_name == "burgers" and model_type in {"deeponet", "pino"}:
        CONFIG["rollout_warmup_epochs"] = 0
        CONFIG["rollout_loss_ramp_epochs"] = 35
        CONFIG["scheduled_sampling_ramp_epochs"] = 60
        CONFIG["residual_limit_multiplier"] = max(float(CONFIG.get("residual_limit_multiplier", 2.0)), 8.0)
        CONFIG["input_noise_warmup_epochs"] = min(int(CONFIG.get("input_noise_warmup_epochs", 60)), 10)
        CONFIG["selection_rollout_warmup_epochs"] = max(int(CONFIG.get("selection_rollout_warmup_epochs", 10)), 35)
        CONFIG["rollout_train_max_batch"] = max(int(CONFIG.get("rollout_train_max_batch", 3)), 12)
        CONFIG["stable_select_weights"] = {"val": 0.02, "raw_rollout": 1.0, "risk": 0.15, "lower_sat": 0.0, "clip_frac": 0.0}
        CONFIG["rollout_select_weights_horizon"] = {40: 0.05, 60: 0.15, 80: 0.45, 100: 0.35}
        # Use a slightly larger rollout curriculum than the first NC12 pass.
        if isinstance(CONFIG.get("rollout_k_max"), dict):
            CONFIG["rollout_k_max"]["burgers"] = max(int(CONFIG["rollout_k_max"].get("burgers", 12)), 16)

    elif pde_name == "grayscott" and model_type == "pinn":
        CONFIG["rollout_warmup_epochs"] = 5
        CONFIG["rollout_loss_ramp_epochs"] = 80
        CONFIG["scheduled_sampling_ramp_epochs"] = 120
        CONFIG["residual_limit_multiplier"] = max(float(CONFIG.get("residual_limit_multiplier", 2.0)), 3.0)
        CONFIG["input_noise_warmup_epochs"] = min(int(CONFIG.get("input_noise_warmup_epochs", 60)), 25)
        CONFIG["selection_rollout_warmup_epochs"] = max(int(CONFIG.get("selection_rollout_warmup_epochs", 10)), 50)
        CONFIG["rollout_train_every"] = min(int(CONFIG.get("rollout_train_every", 4)), 3)
        CONFIG["rollout_train_max_batch"] = max(int(CONFIG.get("rollout_train_max_batch", 3)), 4)
        CONFIG["stable_select_weights"] = {"val": 0.03, "raw_rollout": 1.0, "risk": 0.20, "lower_sat": 0.15, "clip_frac": 0.05}
        CONFIG["rollout_select_weights_horizon"] = {40: 0.10, 60: 0.20, 80: 0.45, 100: 0.25}
        if isinstance(CONFIG.get("rollout_k_max"), dict):
            CONFIG["rollout_k_max"]["grayscott"] = max(int(CONFIG["rollout_k_max"].get("grayscott", 20)), 24)

    elif pde_name == "kolmogorov" and model_type in {"deeponet", "fno"}:
        CONFIG["rollout_warmup_epochs"] = 0
        CONFIG["rollout_loss_ramp_epochs"] = 50
        CONFIG["scheduled_sampling_ramp_epochs"] = 80
        CONFIG["residual_limit_multiplier"] = max(float(CONFIG.get("residual_limit_multiplier", 2.0)), 6.0)
        CONFIG["input_noise_warmup_epochs"] = min(int(CONFIG.get("input_noise_warmup_epochs", 60)), 15)
        CONFIG["selection_rollout_warmup_epochs"] = max(int(CONFIG.get("selection_rollout_warmup_epochs", 10)), 40)
        CONFIG["rollout_train_max_batch"] = max(int(CONFIG.get("rollout_train_max_batch", 3)), 4)
        CONFIG["stable_select_weights"] = {"val": 0.03, "raw_rollout": 1.0, "risk": 0.25, "lower_sat": 0.0, "clip_frac": 0.0}
        CONFIG["rollout_select_weights_horizon"] = {40: 0.05, 60: 0.15, 80: 0.45, 100: 0.35}
        if isinstance(CONFIG.get("rollout_k_max"), dict):
            CONFIG["rollout_k_max"]["kolmogorov"] = max(int(CONFIG["rollout_k_max"].get("kolmogorov", 12)), 16)

    elif pde_name == "kolmogorov" and model_type == "pinn":
        CONFIG["rollout_warmup_epochs"] = 5
        CONFIG["rollout_loss_ramp_epochs"] = 60
        CONFIG["scheduled_sampling_ramp_epochs"] = 100
        CONFIG["residual_limit_multiplier"] = max(float(CONFIG.get("residual_limit_multiplier", 2.0)), 4.0)
        CONFIG["input_noise_warmup_epochs"] = min(int(CONFIG.get("input_noise_warmup_epochs", 60)), 20)
        CONFIG["selection_rollout_warmup_epochs"] = max(int(CONFIG.get("selection_rollout_warmup_epochs", 10)), 45)
        CONFIG["rollout_train_max_batch"] = max(int(CONFIG.get("rollout_train_max_batch", 3)), 4)
        CONFIG["stable_select_weights"] = {"val": 0.04, "raw_rollout": 1.0, "risk": 0.25, "lower_sat": 0.0, "clip_frac": 0.0}
        CONFIG["rollout_select_weights_horizon"] = {40: 0.08, 60: 0.17, 80: 0.45, 100: 0.30}
        if isinstance(CONFIG.get("rollout_k_max"), dict):
            CONFIG["rollout_k_max"]["kolmogorov"] = max(int(CONFIG["rollout_k_max"].get("kolmogorov", 12)), 16)

def smooth_clamp_t(x: torch.Tensor, lo: float = 0.0, hi: float = 1.5, beta: Optional[float] = None) -> torch.Tensor:
    """Differentiable, nearly identity-preserving clamp.

    For x inside [lo, hi], this is close to identity; outside the interval it
    smoothly saturates. This is used as a structural Gray--Scott admissibility
    layer, not as a post-hoc evaluation projection.
    """
    if beta is None:
        beta = float(CONFIG.get("soft_clamp_beta", 25.0))
    beta_t = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
    lo_t = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
    hi_t = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
    return lo_t + F.softplus(beta_t * (x - lo_t)) / beta_t - F.softplus(beta_t * (x - hi_t)) / beta_t


def should_apply_bounded_state(pde_name: str) -> bool:
    return bool(CONFIG.get("grayscott_bounded_state", True)) and pde_name == "grayscott"


def postprocess_next_state_norm(raw_next_norm: torch.Tensor, normalizer: "Normalizer", pde_name: str) -> torch.Tensor:
    """Apply architecture-level differentiable state admissibility if enabled."""
    if should_apply_bounded_state(pde_name):
        raw_phys = normalizer.denorm_state_t(raw_next_norm.float())
        bounded_phys = smooth_clamp_t(raw_phys, 0.0, 1.5)
        return normalizer.norm_state_t(bounded_phys)
    return raw_next_norm


def predict_next_norm(model: nn.Module, model_type: str, pde_name: str,
                      state_norm: torch.Tensor, a_norm: torch.Tensor, normalizer: "Normalizer",
                      x_norm_1d: Optional[torch.Tensor] = None,
                      coords_2d: Optional[torch.Tensor] = None,
                      coord_feat_1d: Optional[torch.Tensor] = None,
                      coord_feat_2d: Optional[torch.Tensor] = None,
                      residual_scale_np: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return next normalized state and the limited normalized residual.

    All training, validation, and rollout paths should use this function so
    bounded-state prediction and residual step limiting are applied consistently.
    """
    delta_raw = predict_delta(model, model_type, pde_name, state_norm, a_norm,
                              x_norm_1d=x_norm_1d, coords_2d=coords_2d,
                              coord_feat_1d=coord_feat_1d, coord_feat_2d=coord_feat_2d)
    delta = apply_residual_step_limit(delta_raw, residual_scale_np)
    raw_next = state_norm + delta
    next_norm = postprocess_next_state_norm(raw_next, normalizer, pde_name)
    return next_norm, delta



def predict_next_norm_with_raw(model: nn.Module, model_type: str, pde_name: str,
                               state_norm: torch.Tensor, a_norm: torch.Tensor, normalizer: "Normalizer",
                               x_norm_1d: Optional[torch.Tensor] = None,
                               coords_2d: Optional[torch.Tensor] = None,
                               coord_feat_1d: Optional[torch.Tensor] = None,
                               coord_feat_2d: Optional[torch.Tensor] = None,
                               residual_scale_np: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return next state, limited residual, and raw residual for repair losses."""
    delta_raw = predict_delta(model, model_type, pde_name, state_norm, a_norm,
                              x_norm_1d=x_norm_1d, coords_2d=coords_2d,
                              coord_feat_1d=coord_feat_1d, coord_feat_2d=coord_feat_2d)
    delta_limited = apply_residual_step_limit(delta_raw, residual_scale_np)
    raw_next = state_norm + delta_limited
    next_norm = postprocess_next_state_norm(raw_next, normalizer, pde_name)
    return next_norm, delta_limited, delta_raw


def increment_fit_loss(delta_raw: torch.Tensor, delta_limited: torch.Tensor,
                       delta_target: torch.Tensor,
                       residual_scale_np: Optional[np.ndarray]) -> torch.Tensor:
    """Scale-normalized residual/increment fitting loss.

    For small-time-step PDEs, next-state MSE can be minimized by an identity-like
    map. This loss forces the model to learn the small normalized increment
    itself, with the data-derived residual scale used for conditioning.
    """
    if residual_scale_np is None:
        return F.mse_loss(delta_limited, delta_target)
    scale = residual_scale_t(residual_scale_np, delta_limited.device, delta_limited.ndim)
    target_s = delta_target / (scale + 1.0e-12)
    raw_s = delta_raw / (scale + 1.0e-12)
    limited_s = delta_limited / (scale + 1.0e-12)
    return 0.75 * F.mse_loss(raw_s, target_s) + 0.25 * F.mse_loss(limited_s, target_s)

def grad_mag_2d_t(z: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    return torch.sqrt(tgradx_2d(z, dx).pow(2) + tgrady_2d(z, dy).pow(2) + 1.0e-12)


def risk_scalar_t(pred_phys: torch.Tensor, pde_name: str, meta: dict) -> torch.Tensor:
    """Differentiable batch risk scalar aligned with risk_scalar_np."""
    if pde_name == "burgers":
        dx = meta["L"] / meta["nx"]
        s = pred_phys[:, 0]
        return torch.amax(torch.abs(tdiff1_1d(s, dx)).reshape(pred_phys.shape[0], -1), dim=1)
    if pde_name == "grayscott":
        dx = meta["Lx"] / meta["nx"]
        dy = meta["Ly"] / meta["ny"]
        v = pred_phys[:, 1]
        return torch.amax(grad_mag_2d_t(v, dx, dy).reshape(pred_phys.shape[0], -1), dim=1)
    if pde_name == "kolmogorov":
        dx = meta["Lx"] / meta["nx"]
        dy = meta["Ly"] / meta["ny"]
        w = pred_phys[:, 0]
        return torch.amax(grad_mag_2d_t(w, dx, dy).reshape(pred_phys.shape[0], -1), dim=1)
    raise ValueError(pde_name)


def rollout_grad_loss_t(pred_phys: torch.Tensor, target_phys: torch.Tensor, pde_name: str, meta: dict) -> torch.Tensor:
    """Gradient-field rollout loss on the risk carrier."""
    if pde_name == "burgers":
        dx = meta["L"] / meta["nx"]
        return F.l1_loss(tdiff1_1d(pred_phys[:, 0], dx), tdiff1_1d(target_phys[:, 0], dx))
    if pde_name == "grayscott":
        dx = meta["Lx"] / meta["nx"]
        dy = meta["Ly"] / meta["ny"]
        return F.l1_loss(grad_mag_2d_t(pred_phys[:, 1], dx, dy), grad_mag_2d_t(target_phys[:, 1], dx, dy))
    if pde_name == "kolmogorov":
        dx = meta["Lx"] / meta["nx"]
        dy = meta["Ly"] / meta["ny"]
        return F.l1_loss(grad_mag_2d_t(pred_phys[:, 0], dx, dy), grad_mag_2d_t(target_phys[:, 0], dx, dy))
    return torch.tensor(0.0, device=pred_phys.device)


def physical_bound_components(pred_phys: torch.Tensor, pde_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if pde_name == "grayscott":
        lower = torch.relu(-pred_phys).pow(2).mean()
        upper = torch.relu(pred_phys - 1.5).pow(2).mean()
        return lower, upper
    z = torch.tensor(0.0, dtype=pred_phys.dtype, device=pred_phys.device)
    return z, z


def get_bound_weights(pde_name: str, epoch: int) -> Tuple[float, float]:
    base_low = float(CONFIG.get("bound_loss_weight_lower", {}).get(pde_name, 0.0))
    base_high = float(CONFIG.get("bound_loss_weight_upper", {}).get(pde_name, 0.0))
    warm = int(CONFIG.get("bound_warmup_epochs", 0))
    if max(base_low, base_high) <= 0.0:
        return 0.0, 0.0
    fac = 1.0 if warm <= 0 else min(1.0, float(epoch + 1) / float(warm))
    return base_low * fac, base_high * fac


def project_phys_state_np(x_phys: np.ndarray, pde_name: str) -> np.ndarray:
    if pde_name == "grayscott":
        return np.clip(x_phys, 0.0, 1.5)
    return x_phys


def projection_diagnostics_np(raw_phys: np.ndarray, proj_phys: np.ndarray, pde_name: str) -> Dict[str, float]:
    if pde_name != "grayscott":
        return {"clip_frac": 0.0, "upper_saturation_frac": 0.0, "lower_saturation_frac": 0.0, "projection_rel_l2": 0.0}
    eps = float(CONFIG.get("projection_clip_eps", 1.0e-6))
    raw = raw_phys.astype(np.float64)
    proj = proj_phys.astype(np.float64)
    clip_mask = (raw < 0.0) | (raw > 1.5)
    upper = proj >= (1.5 - eps)
    lower = proj <= eps
    rel = float(np.linalg.norm((proj - raw).reshape(-1)) / (np.linalg.norm(raw.reshape(-1)) + 1e-12))
    return {
        "clip_frac": float(np.mean(clip_mask)),
        "upper_saturation_frac": float(np.mean(upper)),
        "lower_saturation_frac": float(np.mean(lower)),
        "projection_rel_l2": rel,
    }

# ============================================================
# Finite difference helpers
# ============================================================
def tdiff1_1d(u: torch.Tensor, dx: float):
    return (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2.0 * dx)


def tdiff2_1d(u: torch.Tensor, dx: float):
    return (torch.roll(u, -1, dims=-1) - 2.0 * u + torch.roll(u, 1, dims=-1)) / (dx ** 2)


def tgradx_2d(z: torch.Tensor, dx: float):
    return (torch.roll(z, -1, dims=-1) - torch.roll(z, 1, dims=-1)) / (2.0 * dx)


def tgrady_2d(z: torch.Tensor, dy: float):
    return (torch.roll(z, -1, dims=-2) - torch.roll(z, 1, dims=-2)) / (2.0 * dy)


def tlaplacian_2d(z: torch.Tensor, dx: float, dy: float):
    return (
        (torch.roll(z, -1, dims=-1) - 2.0 * z + torch.roll(z, 1, dims=-1)) / (dx ** 2)
        + (torch.roll(z, -1, dims=-2) - 2.0 * z + torch.roll(z, 1, dims=-2)) / (dy ** 2)
    )


# ============================================================
# Fourier features for coordinate decoders
# ============================================================
def fourier_features_1d(x: torch.Tensor, freqs: List[int]) -> torch.Tensor:
    # x: [N] in [-1,1] or [B,N,1]
    if x.ndim == 1:
        base = x[:, None]
    elif x.ndim == 3:
        base = x
    else:
        raise ValueError(x.shape)
    feats = [base]
    for f in freqs:
        feats.append(torch.sin(math.pi * f * base))
        feats.append(torch.cos(math.pi * f * base))
    return torch.cat(feats, dim=-1)


def fourier_features_2d(coords: torch.Tensor, freqs: List[int]) -> torch.Tensor:
    # coords: [H,W,2] or [B,H,W,2]
    x = coords[..., 0:1]
    y = coords[..., 1:2]
    feats = [x, y]
    for f in freqs:
        feats += [torch.sin(math.pi * f * x), torch.cos(math.pi * f * x),
                  torch.sin(math.pi * f * y), torch.cos(math.pi * f * y)]
    return torch.cat(feats, dim=-1)


def coord_dim_1d(freqs: List[int]) -> int:
    return 1 + 2 * len(freqs)


def coord_dim_2d(freqs: List[int]) -> int:
    return 2 + 4 * len(freqs)


# ============================================================
# FNO models
# ============================================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        # cuFFT does not support non-power-of-two FFT sizes in fp16/bf16 on some
        # CUDA paths (e.g. 48x48 Gray--Scott under AMP). Keep the spectral FFT
        # branch in fp32 while allowing the rest of the model to use AMP.
        out_dtype = x.dtype
        B, C, N = x.shape
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            x_ft = rfft(x32, dim=-1)
            out_ft = torch.zeros(B, self.out_channels, N // 2 + 1, device=x.device, dtype=torch.cfloat)
            m = min(self.modes, x_ft.shape[-1])
            out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m])
            out = irfft(out_ft, n=N, dim=-1)
        return out.to(out_dtype)


class FNO1d(nn.Module):
    def __init__(self, in_channels, out_channels, width=32, modes=16, n_layers=4):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.specs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, out_channels, 1))

    def forward(self, x):
        x = self.lift(x)
        for spec, w in zip(self.specs, self.ws):
            x = F.gelu(spec(x) + w(x))
        return self.proj(x)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        # cuFFT half precision only supports power-of-two sizes. Gray--Scott and
        # Kolmogorov use 48x48 grids, so AMP can otherwise crash at rfft2.
        # We isolate only the FFT branch in fp32; Conv/GELU layers still benefit
        # from AMP outside this block.
        out_dtype = x.dtype
        B, C, H, W = x.shape
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            x_ft = rfft2(x32, dim=(-2, -1))
            out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)
            m1 = min(self.modes1, x_ft.shape[-2])
            m2 = min(self.modes2, x_ft.shape[-1])
            out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weight[:, :, :m1, :m2])
            out = irfft2(out_ft, s=(H, W), dim=(-2, -1))
        return out.to(out_dtype)


class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width=32, modes=16, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.specs = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1))

    def forward(self, x):
        x = self.lift(x)
        for spec, w in zip(self.specs, self.ws):
            x = F.gelu(spec(x) + w(x))
        return self.proj(x)


# ============================================================
# CNN encoders and improved DeepONet/PINN
# ============================================================
class CNNEncoder1d(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 96, 5, padding=2), nn.GELU(),
            nn.Conv1d(96, 128, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, latent_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CNNEncoder2d(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1), nn.GELU(),
            nn.Conv2d(48, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 96, 3, padding=1), nn.GELU(),
            nn.Conv2d(96, 128, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, latent_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class LatentPINN1d(nn.Module):
    """CNN encoder + Fourier-coordinate decoder. Predicts normalized residual field."""
    def __init__(self, in_channels: int, out_channels: int, latent_dim: int, coord_dim: int, hidden: int, depth: int):
        super().__init__()
        self.encoder = CNNEncoder1d(in_channels, latent_dim)
        dims = [latent_dim + coord_dim] + [hidden] * depth + [out_channels]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.decoder = nn.Sequential(*layers)

    def forward(self, field_input: torch.Tensor, coord_feat: torch.Tensor):
        # field_input [B,C+1,N], coord_feat [N,F]
        B = field_input.shape[0]
        z = self.encoder(field_input)  # [B,L]
        N = coord_feat.shape[0]
        cf = coord_feat[None, :, :].repeat(B, 1, 1)
        zz = z[:, None, :].repeat(1, N, 1)
        out = self.decoder(torch.cat([zz, cf], dim=-1))  # [B,N,C]
        return out.permute(0, 2, 1).contiguous()


class LatentPINN2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, latent_dim: int, coord_dim: int, hidden: int, depth: int):
        super().__init__()
        self.encoder = CNNEncoder2d(in_channels, latent_dim)
        self.out_channels = out_channels
        dims = [latent_dim + coord_dim] + [hidden] * depth + [out_channels]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.decoder = nn.Sequential(*layers)

    def forward(self, field_input: torch.Tensor, coord_feat: torch.Tensor):
        # field_input [B,C+1,H,W], coord_feat [H,W,F]
        B, _, H, W = field_input.shape
        z = self.encoder(field_input)
        cf = coord_feat.view(1, H, W, -1).repeat(B, 1, 1, 1)
        zz = z[:, None, None, :].repeat(1, H, W, 1)
        out = self.decoder(torch.cat([zz, cf], dim=-1))  # [B,H,W,C]
        return out.permute(0, 3, 1, 2).contiguous()


class CNNDeepONet1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, latent_dim: int, coord_dim: int, hidden: int):
        super().__init__()
        self.out_channels = out_channels
        self.hidden = hidden
        self.encoder = CNNEncoder1d(in_channels, latent_dim)
        self.branch = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden * out_channels))
        self.trunk = nn.Sequential(nn.Linear(coord_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.use_local_refine = bool(CONFIG.get("deeponet_local_refine_1d", True))
        if self.use_local_refine:
            width = int(CONFIG.get("deeponet_local_refine_width_1d", 64))
            self.local_refine = nn.Sequential(
                nn.Conv1d(in_channels + out_channels, width, 5, padding=2), nn.GELU(),
                nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
                nn.Conv1d(width, out_channels, 5, padding=2),
            )
            nn.init.zeros_(self.local_refine[-1].weight)
            nn.init.zeros_(self.local_refine[-1].bias)
            self._scale_raw = nn.Parameter(torch.tensor(float(CONFIG.get("deeponet_local_refine_scale_init_1d", 0.0))))
        else:
            self.local_refine = None

    def forward(self, field_input: torch.Tensor, coord_feat: torch.Tensor):
        B = field_input.shape[0]
        z = self.encoder(field_input)
        b = self.branch(z).view(B, self.out_channels, self.hidden)
        t = self.trunk(coord_feat)[None, :, :].repeat(B, 1, 1)
        out = torch.einsum("bch,bnh->bcn", b, t) + self.bias[None, :, None]
        if self.local_refine is not None:
            max_scale = float(CONFIG.get("deeponet_local_refine_scale_max_1d", 0.30))
            scale = max_scale * torch.tanh(self._scale_raw)
            out = out + scale * self.local_refine(torch.cat([field_input, out], dim=1))
        return out


class CNNDeepONet2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, latent_dim: int, coord_dim: int, hidden: int):
        super().__init__()
        self.out_channels = out_channels
        self.hidden = hidden
        self.encoder = CNNEncoder2d(in_channels, latent_dim)
        self.branch = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden * out_channels))
        self.trunk = nn.Sequential(nn.Linear(coord_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.use_local_refine = bool(CONFIG.get("deeponet_local_refine", True))
        if self.use_local_refine:
            # Small local correction head to preserve reaction--diffusion fronts.
            self.local_refine = nn.Sequential(
                nn.Conv2d(in_channels + out_channels, 48, 3, padding=1), nn.GELU(),
                nn.Conv2d(48, 48, 3, padding=1), nn.GELU(),
                nn.Conv2d(48, out_channels, 3, padding=1),
            )
            # Start as nearly pure DeepONet; let training learn local corrections.
            nn.init.zeros_(self.local_refine[-1].weight)
            nn.init.zeros_(self.local_refine[-1].bias)
        else:
            self.local_refine = None

    def forward(self, field_input: torch.Tensor, coord_feat: torch.Tensor):
        B, _, H, W = field_input.shape
        z = self.encoder(field_input)
        b = self.branch(z).view(B, self.out_channels, self.hidden)
        t = self.trunk(coord_feat.view(H * W, -1))[None, :, :].repeat(B, 1, 1)
        out = torch.einsum("bch,bnh->bcn", b, t) + self.bias[None, :, None]
        out = out.view(B, self.out_channels, H, W)
        if self.local_refine is not None:
            out = out + self.local_refine(torch.cat([field_input, out], dim=1))
        return out


class UniversalRefinedModel1d(nn.Module):
    """Model-agnostic local correction wrapper for 1D surrogates.

    Zero initialization makes the initial wrapper exactly equal to the base
    model. It is used for Burgers FNO/PINO/PINN where small phase and gradient
    residual errors accumulate over long autoregressive rollouts.
    """
    def __init__(self, base: nn.Module, in_channels: int, out_channels: int, width: Optional[int] = None):
        super().__init__()
        self.base = base
        width = int(width or CONFIG.get("local_refine_width_1d", 64))
        self.refine = nn.Sequential(
            nn.Conv1d(in_channels + out_channels, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, out_channels, 5, padding=2),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        self._scale_raw = nn.Parameter(torch.tensor(float(CONFIG.get("local_refine_scale_init", 0.0))))

    def forward(self, *args):
        out = self.base(*args)
        field_input = args[0]
        max_scale = float(CONFIG.get("local_refine_scale_max_1d", 0.30))
        scale = max_scale * torch.tanh(self._scale_raw)
        return out + scale * self.refine(torch.cat([field_input, out], dim=1))


class UniversalRefinedModel2d(nn.Module):
    """Model-agnostic local correction wrapper for 2D surrogates.

    It is zero-initialized and bounded by a small learnable scale, so the
    untrained wrapper behaves exactly like the base model. During training it
    learns local reaction-front corrections for FNO/PINO/PINN Gray--Scott models.
    """
    def __init__(self, base: nn.Module, in_channels: int, out_channels: int, width: Optional[int] = None):
        super().__init__()
        self.base = base
        width = int(width or CONFIG.get("local_refine_width_2d", 48))
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        self._scale_raw = nn.Parameter(torch.tensor(float(CONFIG.get("local_refine_scale_init", 0.0))))

    def forward(self, *args):
        out = self.base(*args)
        field_input = args[0]
        max_scale = float(CONFIG.get("local_refine_scale_max", 0.35))
        scale = max_scale * torch.tanh(self._scale_raw)
        return out + scale * self.refine(torch.cat([field_input, out], dim=1))


# ============================================================
# Model factory and prediction
# ============================================================
def get_model(model_type: str, pde_name: str, state_shape: Tuple[int, ...], coord_feature_dim: int):
    if pde_name == "burgers":
        C, N = state_shape
        in_op = C + 2  # state + normalized control + x
        in_cnn = C + 1  # state + normalized control plane
        if model_type in ["fno", "pino"]:
            base = FNO1d(
                in_op, C,
                cfg_dim("fno_width_1d", pde_name, model_type),
                cfg_dim("fno_modes_1d", pde_name, model_type),
                CONFIG["fno_layers_1d"],
            )
            if is_nc12boost_active(pde_name, model_type) and bool(CONFIG.get("universal_local_refine_1d", True)):
                if model_type in set(CONFIG.get("universal_local_refine_1d_models", [])):
                    base = UniversalRefinedModel1d(base, in_op, C)
            return base
        if model_type == "pinn":
            base = LatentPINN1d(
                in_cnn, C,
                int(CONFIG["pinn_latent_dim_1d_boost"]) if is_nc12boost_active(pde_name, model_type) else int(CONFIG["latent_dim_1d"]),
                coord_feature_dim,
                int(CONFIG["pinn_coord_hidden_1d_boost"]) if is_nc12boost_active(pde_name, model_type) else int(CONFIG["coord_hidden_1d"]),
                int(CONFIG["pinn_coord_depth_1d_boost"]) if is_nc12boost_active(pde_name, model_type) else int(CONFIG["coord_depth_1d"]),
            )
            if is_nc12boost_active(pde_name, model_type) and bool(CONFIG.get("pinn_local_refine_1d", True)):
                base = UniversalRefinedModel1d(base, in_cnn, C)
            return base
        if model_type == "deeponet":
            return CNNDeepONet1d(
                in_cnn, C,
                int(CONFIG["deeponet_latent_dim_1d_boost"]) if is_nc12boost_active(pde_name, model_type) else int(CONFIG["latent_dim_1d"]),
                coord_feature_dim,
                cfg_dim("deeponet_hidden_1d", pde_name, model_type),
            )
    else:
        C, H, W = state_shape
        in_op = C + 3  # state + normalized control + x + y
        in_cnn = C + 1
        base: nn.Module
        refine_in_channels: int
        if model_type in ["fno", "pino"]:
            base = FNO2d(
                in_op, C,
                cfg_dim("fno_width_2d", pde_name, model_type),
                cfg_dim("fno_modes_2d", pde_name, model_type),
                CONFIG["fno_layers_2d"],
            )
            refine_in_channels = in_op
        elif model_type == "pinn":
            base = LatentPINN2d(
                in_cnn, C,
                int(CONFIG["pinn_latent_dim_2d_boost"]) if is_nc12boost_active(pde_name, model_type) else int(CONFIG["latent_dim_2d"]),
                coord_feature_dim,
                int(CONFIG["pinn_coord_hidden_2d_boost"]) if is_nc12boost_active(pde_name, model_type) else int(CONFIG["coord_hidden_2d"]),
                int(CONFIG["pinn_coord_depth_2d_boost"]) if is_nc12boost_active(pde_name, model_type) else int(CONFIG["coord_depth_2d"]),
            )
            refine_in_channels = in_cnn
        elif model_type == "deeponet":
            base = CNNDeepONet2d(
                in_cnn, C,
                int(CONFIG["deeponet_latent_dim_2d_boost"]) if is_nc12boost_active(pde_name, model_type) else int(CONFIG["latent_dim_2d"]),
                coord_feature_dim,
                cfg_dim("deeponet_hidden_2d", pde_name, model_type),
            )
            refine_in_channels = in_cnn
        else:
            raise ValueError(f"Unsupported model_type={model_type}, pde_name={pde_name}")

        if is_protected_grayscott_model(pde_name, model_type):
            # Keep already validated Gray--Scott DeepONet/PINO/FNO paths exactly as in the improved script.
            refine_models = set(CONFIG.get("universal_local_refine_models_2d", []))
            if (pde_name == "grayscott" and bool(CONFIG.get("universal_local_refine_2d", True)) and model_type in refine_models):
                base = UniversalRefinedModel2d(base, refine_in_channels, C)
            return base

        if is_nc12boost_active(pde_name, model_type):
            if model_type == "pinn" and bool(CONFIG.get("pinn_local_refine_2d", True)):
                base = UniversalRefinedModel2d(base, refine_in_channels, C, width=int(CONFIG.get("pinn_local_refine_width_2d", 64)))
            elif model_type in ["fno", "pino"] and bool(CONFIG.get("universal_local_refine_2d", True)):
                if model_type in set(CONFIG.get("universal_local_refine_2d_models_boost", [])):
                    base = UniversalRefinedModel2d(base, refine_in_channels, C)
        else:
            refine_models = set(CONFIG.get("universal_local_refine_models_2d", []))
            if (pde_name == "grayscott" and bool(CONFIG.get("universal_local_refine_2d", True)) and model_type in refine_models):
                base = UniversalRefinedModel2d(base, refine_in_channels, C)
        return base
    raise ValueError(f"Unsupported model_type={model_type}, pde_name={pde_name}")


def make_field_input_with_control(state_norm: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
    if a_norm.ndim == 1:
        a_norm = a_norm[:, None]
    if state_norm.ndim == 3:  # [B,C,N]
        B, _, N = state_norm.shape
        ctrl = a_norm[:, :, None].repeat(1, 1, N)
        return torch.cat([state_norm, ctrl], dim=1)
    if state_norm.ndim == 4:
        B, _, H, W = state_norm.shape
        ctrl = a_norm[:, :, None, None].repeat(1, 1, H, W)
        return torch.cat([state_norm, ctrl], dim=1)
    raise ValueError(state_norm.shape)


def predict_delta(model: nn.Module, model_type: str, pde_name: str,
                  state_norm: torch.Tensor, a_norm: torch.Tensor,
                  x_norm_1d: Optional[torch.Tensor] = None,
                  coords_2d: Optional[torch.Tensor] = None,
                  coord_feat_1d: Optional[torch.Tensor] = None,
                  coord_feat_2d: Optional[torch.Tensor] = None) -> torch.Tensor:
    if a_norm.ndim == 1:
        a_norm = a_norm[:, None]
    if pde_name == "burgers":
        if state_norm.ndim == 2:
            state_norm = state_norm[:, None, :]
        B, C, N = state_norm.shape
        if model_type in ["fno", "pino"]:
            ctrl = a_norm[:, :, None].repeat(1, 1, N)
            xplane = x_norm_1d[None, None, :].repeat(B, 1, 1)
            inp = torch.cat([state_norm, ctrl, xplane], dim=1)
            return model(inp)
        else:
            field_input = make_field_input_with_control(state_norm, a_norm)
            return model(field_input, coord_feat_1d)
    else:
        if state_norm.ndim == 3:
            state_norm = state_norm[None]
        B, C, H, W = state_norm.shape
        if model_type in ["fno", "pino"]:
            ctrl = a_norm[:, :, None, None].repeat(1, 1, H, W)
            xy = coords_2d.permute(2, 0, 1)[None, :, :, :].repeat(B, 1, 1, 1)
            inp = torch.cat([state_norm, ctrl, xy], dim=1)
            return model(inp)
        else:
            field_input = make_field_input_with_control(state_norm, a_norm)
            return model(field_input, coord_feat_2d)


# ============================================================
# Physics residuals on physical states
# ============================================================
def burgers_physics_loss(pred_next: torch.Tensor, state_t: torch.Tensor, a_t_phys: torch.Tensor,
                         actuator: torch.Tensor, dt: float, dx: float, nu: float):
    u1 = pred_next[:, 0, :]
    u0 = state_t[:, 0, :]
    ux = tdiff1_1d(u1, dx)
    uxx = tdiff2_1d(u1, dx)
    res = (u1 - u0) / dt + u1 * ux - nu * uxx - a_t_phys * actuator[None, :]
    return torch.mean(res ** 2)


def grayscott_physics_loss(pred_next: torch.Tensor, state_t: torch.Tensor, a_t_phys: torch.Tensor,
                           actuator: torch.Tensor, dt: float, dx: float, dy: float,
                           Du: float, Dv: float, F0: float, k0: float):
    u1 = pred_next[:, 0]
    v1 = pred_next[:, 1]
    u0 = state_t[:, 0]
    v0 = state_t[:, 1]
    Lu = tlaplacian_2d(u1, dx, dy)
    Lv = tlaplacian_2d(v1, dx, dy)
    uvv = u1 * (v1 ** 2)
    ru = (u1 - u0) / dt - (Du * Lu - uvv + F0 * (1.0 - u1))
    rv = (v1 - v0) / dt - (Dv * Lv + uvv - (F0 + k0) * v1 + a_t_phys[:, :, None] * actuator[None, :, :])
    return torch.mean(ru ** 2) + torch.mean(rv ** 2)


def kolmogorov_physics_loss(pred_next: torch.Tensor, state_t: torch.Tensor, a_t_phys: torch.Tensor,
                            actuator: torch.Tensor, dt: float, dx: float, dy: float,
                            nu: float, forcing: torch.Tensor):
    w1 = pred_next[:, 0]
    w0 = state_t[:, 0]
    B, H, W = w1.shape
    w_hat = torch.fft.fft2(w1, dim=(-2, -1))
    kx = 2.0 * np.pi * torch.fft.fftfreq(W, d=dx, device=w1.device)
    ky = 2.0 * np.pi * torch.fft.fftfreq(H, d=dy, device=w1.device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    invK2 = torch.zeros_like(K2)
    mask = K2 > 0
    invK2[mask] = 1.0 / K2[mask]
    psi_hat = w_hat * invK2[None, :, :]
    psi_hat[:, 0, 0] = 0.0 + 0.0j
    u = torch.fft.ifft2(1j * KY[None, :, :] * psi_hat, dim=(-2, -1)).real
    v = torch.fft.ifft2(-1j * KX[None, :, :] * psi_hat, dim=(-2, -1)).real
    wx = tgradx_2d(w1, dx)
    wy = tgrady_2d(w1, dy)
    lap = tlaplacian_2d(w1, dx, dy)
    res = (w1 - w0) / dt + u * wx + v * wy - nu * lap - forcing[None, :, :] - a_t_phys[:, :, None] * actuator[None, :, :]
    return torch.mean(res ** 2)


def get_base_physics_weight(model_type: str, pde_name: str, epoch: int) -> float:
    if model_type not in ["pinn", "pino"]:
        return 0.0
    base = CONFIG["physics_loss_weight"].get(pde_name, {}).get(model_type, 0.0)
    warm = CONFIG["physics_warmup_epochs"]
    if warm <= 0:
        return base
    return base * min(1.0, float(epoch + 1) / float(warm))


def gradient_risk_loss(pred_phys: torch.Tensor, target_phys: torch.Tensor, pde_name: str, meta: dict) -> torch.Tensor:
    if pde_name == "burgers":
        dx = meta["L"] / meta["nx"]
        gp = tdiff1_1d(pred_phys[:, 0], dx)
        gt = tdiff1_1d(target_phys[:, 0], dx)
        return F.l1_loss(gp, gt)
    if pde_name == "grayscott":
        dx = meta["Lx"] / meta["nx"]
        dy = meta["Ly"] / meta["ny"]
        # risk carrier is v channel
        vp = pred_phys[:, 1]
        vt = target_phys[:, 1]
        gp = torch.sqrt(tgradx_2d(vp, dx) ** 2 + tgrady_2d(vp, dy) ** 2 + 1e-12)
        gt = torch.sqrt(tgradx_2d(vt, dx) ** 2 + tgrady_2d(vt, dy) ** 2 + 1e-12)
        return F.l1_loss(gp, gt)
    if pde_name == "kolmogorov":
        dx = meta["Lx"] / meta["nx"]
        dy = meta["Ly"] / meta["ny"]
        wp = pred_phys[:, 0]
        wt = target_phys[:, 0]
        gp = torch.sqrt(tgradx_2d(wp, dx) ** 2 + tgrady_2d(wp, dy) ** 2 + 1e-12)
        gt = torch.sqrt(tgradx_2d(wt, dx) ** 2 + tgrady_2d(wt, dy) ** 2 + 1e-12)
        return F.l1_loss(gp, gt)
    return torch.tensor(0.0, device=pred_phys.device)


def physics_loss_dispatch(pde_name: str, pred_phys: torch.Tensor, curr_phys: torch.Tensor, a_phys: torch.Tensor,
                          actuator_t: torch.Tensor, meta: dict, forcing_t: Optional[torch.Tensor] = None) -> torch.Tensor:
    if pde_name == "burgers":
        dx = meta["L"] / meta["nx"]
        return burgers_physics_loss(pred_phys, curr_phys, a_phys, actuator_t, meta["dt"], dx, meta["nu"])
    if pde_name == "grayscott":
        dx = meta["Lx"] / meta["nx"]
        dy = meta["Ly"] / meta["ny"]
        return grayscott_physics_loss(pred_phys, curr_phys, a_phys, actuator_t, meta["dt"], dx, dy, meta["Du"], meta["Dv"], meta["F"], meta["k"])
    if pde_name == "kolmogorov":
        dx = meta["Lx"] / meta["nx"]
        dy = meta["Ly"] / meta["ny"]
        assert forcing_t is not None
        return kolmogorov_physics_loss(pred_phys, curr_phys, a_phys, actuator_t, meta["dt"], dx, dy, meta["nu"], forcing_t)
    raise ValueError(pde_name)


# ============================================================
# Risk and rollout evaluation
# ============================================================
def risk_scalar_np(state_phys: np.ndarray, pde_name: str, meta: dict) -> float:
    if pde_name == "burgers":
        s = state_phys[0] if state_phys.ndim == 2 else state_phys
        dx = meta["L"] / meta["nx"]
        gx = (np.roll(s, -1) - np.roll(s, 1)) / (2.0 * dx)
        return float(np.max(np.abs(gx)))
    if pde_name == "grayscott":
        v = state_phys[1]
        dx = meta["Lx"] / meta["nx"]
        dy = meta["Ly"] / meta["ny"]
        gx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dx)
        gy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dy)
        return float(np.max(np.sqrt(gx**2 + gy**2)))
    if pde_name == "kolmogorov":
        w = state_phys[0]
        dx = meta["Lx"] / meta["nx"]
        dy = meta["Ly"] / meta["ny"]
        gx = (np.roll(w, -1, axis=1) - np.roll(w, 1, axis=1)) / (2.0 * dx)
        gy = (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) / (2.0 * dy)
        return float(np.max(np.sqrt(gx**2 + gy**2)))
    raise ValueError(pde_name)


def rollout_model_np(model: nn.Module, model_type: str, pde_name: str, s0_phys: np.ndarray, controls_phys: np.ndarray,
                     normalizer: Normalizer, x_norm_1d: Optional[torch.Tensor], coords_2d: Optional[torch.Tensor],
                     coord_feat_1d: Optional[torch.Tensor], coord_feat_2d: Optional[torch.Tensor],
                     residual_scale_np: Optional[np.ndarray] = None,
                     projected: bool = False,
                     return_diag: bool = False):
    """Autoregressive rollout.

    projected=False is the honest raw neural rollout.
    projected=True is diagnostic only: it applies PDE-specific admissible-state
    projection and records projection burden.
    """
    model.eval()
    states = []
    diag_acc = {"clip_frac": [], "upper_saturation_frac": [], "lower_saturation_frac": [], "projection_rel_l2": []}

    s_norm_np = normalizer.normalize_state_np(s0_phys)
    curr = torch.from_numpy(s_norm_np[None].astype(np.float32)).to(DEVICE)
    states.append(normalizer.denorm_state_t(curr).detach().cpu().numpy()[0])

    with torch.no_grad():
        for a in controls_phys:
            a_phys = torch.tensor([[float(a)]], dtype=torch.float32, device=DEVICE)
            a_norm = normalizer.norm_control_t(a_phys)
            curr_raw, _ = predict_next_norm(
                model, model_type, pde_name, curr, a_norm, normalizer,
                x_norm_1d=x_norm_1d.to(DEVICE) if x_norm_1d is not None else None,
                coords_2d=coords_2d.to(DEVICE) if coords_2d is not None else None,
                coord_feat_1d=coord_feat_1d.to(DEVICE) if coord_feat_1d is not None else None,
                coord_feat_2d=coord_feat_2d.to(DEVICE) if coord_feat_2d is not None else None,
                residual_scale_np=residual_scale_np,
            )
            curr_phys_raw = normalizer.denorm_state_t(curr_raw).detach().cpu().numpy()[0]

            if projected:
                curr_phys_proj = project_phys_state_np(curr_phys_raw, pde_name)
                d = projection_diagnostics_np(curr_phys_raw, curr_phys_proj, pde_name)
                for k in diag_acc:
                    diag_acc[k].append(d[k])
                curr = normalizer.norm_state_t(torch.from_numpy(curr_phys_proj[None].astype(np.float32)).to(DEVICE))
                states.append(curr_phys_proj.astype(np.float32))
            else:
                curr = curr_raw
                states.append(curr_phys_raw.astype(np.float32))

            if not np.isfinite(states[-1]).all():
                break

    arr = np.stack(states, axis=0).astype(np.float32)
    diag = {k + "_mean": float(np.mean(v)) if len(v) else 0.0 for k, v in diag_acc.items()}
    diag.update({k + "_max": float(np.max(v)) if len(v) else 0.0 for k, v in diag_acc.items()})
    if return_diag:
        return arr, diag
    return arr


# ============================================================
# Loss helpers
# ============================================================
def compute_rollout_loss(model, model_type, pde_name, dataset: NormalizedPairDataset,
                         tr_idx: torch.Tensor, t_idx: torch.Tensor, state0_norm: torch.Tensor,
                         normalizer: Normalizer, x_norm_1d, coords_2d, coord_feat_1d, coord_feat_2d,
                         K: int, residual_scale_np: Optional[np.ndarray] = None,
                         meta: Optional[dict] = None, epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-conscious truncated rollout loss.

    Returns state rollout loss, physical-bound loss, risk-rollout loss, and
    gradient-rollout loss. Scheduled sampling is used so the model learns to
    recover from its own off-manifold predictions.
    """
    if K <= 0:
        z = torch.tensor(0.0, device=state0_norm.device)
        return z, z, z, z

    curr = state0_norm
    state_losses: List[torch.Tensor] = []
    bound_losses: List[torch.Tensor] = []
    risk_losses: List[torch.Tensor] = []
    grad_losses: List[torch.Tensor] = []
    tr_np = tr_idx.detach().cpu().numpy().astype(int)
    t_np = t_idx.detach().cpu().numpy().astype(int)

    ss_max = float(CONFIG.get("scheduled_sampling_max", 0.0))
    ramp = max(1, int(CONFIG.get("scheduled_sampling_ramp_epochs", 1)))
    warm = int(CONFIG.get("rollout_warmup_epochs", 0))
    ss_prob = ss_max * min(1.0, max(0.0, float(epoch - warm + 1) / float(ramp)))

    for k in range(K):
        valid = (t_np + k + 1) <= dataset.T
        if not np.any(valid):
            break
        ids = np.where(valid)[0]
        curr_v = curr[ids]
        tr_v = tr_np[ids]
        tt_v = t_np[ids] + k

        a_phys_np = dataset.control[tr_v, tt_v].astype(np.float32)[:, None]
        a_norm_np = dataset.normalizer.normalize_control_np(a_phys_np)
        target_phys_np = dataset.state[tr_v, tt_v + 1].astype(np.float32)
        target_np = np.stack([dataset.normalizer.normalize_state_np(x) for x in target_phys_np], axis=0)

        a_norm = torch.from_numpy(a_norm_np).to(state0_norm.device)
        target = torch.from_numpy(target_np).to(state0_norm.device)

        pred_next, _ = predict_next_norm(model, model_type, pde_name, curr_v, a_norm, normalizer,
                                         x_norm_1d=x_norm_1d, coords_2d=coords_2d,
                                         coord_feat_1d=coord_feat_1d, coord_feat_2d=coord_feat_2d,
                                         residual_scale_np=residual_scale_np)
        state_losses.append(F.mse_loss(pred_next, target))

        pred_next_phys = normalizer.denorm_state_t(pred_next.float())
        target_phys = normalizer.denorm_state_t(target.float())
        bound_low_k, bound_high_k = physical_bound_components(pred_next_phys, pde_name)
        bound_losses.append(bound_low_k + bound_high_k)

        if meta is not None:
            risk_losses.append(F.l1_loss(risk_scalar_t(pred_next_phys, pde_name, meta),
                                         risk_scalar_t(target_phys, pde_name, meta)))
            grad_losses.append(rollout_grad_loss_t(pred_next_phys, target_phys, pde_name, meta))

        if ss_prob > 0.0 and k < K - 1:
            if curr_v.ndim == 4:
                mask_shape = (len(ids), 1, 1, 1)
            else:
                mask_shape = (len(ids), 1, 1)
            use_pred = (torch.rand(mask_shape, device=state0_norm.device) < ss_prob).float()
            next_in = use_pred * pred_next.detach() + (1.0 - use_pred) * target
        else:
            next_in = pred_next

        curr_new = curr.clone()
        curr_new[ids] = next_in
        curr = curr_new

    z = torch.tensor(0.0, device=state0_norm.device)
    if not state_losses:
        return z, z, z, z
    state_l = torch.stack(state_losses).mean()
    bound_l = torch.stack(bound_losses).mean() if bound_losses else z
    risk_l = torch.stack(risk_losses).mean() if risk_losses else z
    grad_l = torch.stack(grad_losses).mean() if grad_losses else z
    return state_l, bound_l, risk_l, grad_l


# ============================================================
# Evaluation helpers
# ============================================================
def eval_one_step(model: nn.Module, model_type: str, pde_name: str, loader: DataLoader,
                  normalizer: Normalizer, x_norm_1d=None, coords_2d=None, coord_feat_1d=None, coord_feat_2d=None,
                  residual_scale_np: Optional[np.ndarray] = None):
    model.eval()
    rel_l2_list, mae_list = [], []
    risk_pred, risk_true = [], []
    with torch.no_grad():
        for batch in loader:
            s0_n, delta_n, s1_n, a_n, a_phys, tr, tt = batch
            s0_n = s0_n.to(DEVICE); delta_n = delta_n.to(DEVICE); s1_n = s1_n.to(DEVICE); a_n = a_n.to(DEVICE)
            pred_next_n, _ = predict_next_norm(
                model, model_type, pde_name, s0_n, a_n, normalizer,
                x_norm_1d=x_norm_1d.to(DEVICE) if x_norm_1d is not None else None,
                coords_2d=coords_2d.to(DEVICE) if coords_2d is not None else None,
                coord_feat_1d=coord_feat_1d.to(DEVICE) if coord_feat_1d is not None else None,
                coord_feat_2d=coord_feat_2d.to(DEVICE) if coord_feat_2d is not None else None,
                residual_scale_np=residual_scale_np,
            )
            pred_phys = normalizer.denorm_state_t(pred_next_n)
            true_phys = normalizer.denorm_state_t(s1_n)
            diff = pred_phys - true_phys
            B = diff.shape[0]
            rel = torch.norm(diff.reshape(B, -1), dim=1) / (torch.norm(true_phys.reshape(B, -1), dim=1) + 1e-12)
            mae = torch.mean(torch.abs(diff).reshape(B, -1), dim=1)
            rel_l2_list.append(rel.cpu().numpy())
            mae_list.append(mae.cpu().numpy())
            # risk metrics on CPU for a subset of batches (all is fine for 20k samples still okay, but limit CPU overhead slightly)
            pred_np = pred_phys.detach().cpu().numpy()
            true_np = true_phys.detach().cpu().numpy()
            for i in range(B):
                risk_pred.append(risk_scalar_np(pred_np[i], pde_name, loader.dataset.normalizer.stats.__dict__.get('meta', {}) if False else loader.dataset_meta if hasattr(loader, 'dataset_meta') else {}))
                risk_true.append(0.0)
    rel_all = np.concatenate(rel_l2_list)
    mae_all = np.concatenate(mae_list)
    return float(rel_all.mean()), float(rel_all.std()), float(mae_all.mean()), float(mae_all.std()), rel_all, mae_all


def eval_one_step_with_risk(model: nn.Module, model_type: str, pde_name: str, loader: DataLoader,
                            normalizer: Normalizer, meta: dict,
                            x_norm_1d=None, coords_2d=None, coord_feat_1d=None, coord_feat_2d=None,
                            residual_scale_np: Optional[np.ndarray] = None,
                            max_risk_samples: int = 2000):
    model.eval()
    rel_l2_list, mae_list = [], []
    risk_pred, risk_true = [], []
    n_risk = 0
    with torch.no_grad():
        for batch in loader:
            s0_n, delta_n, s1_n, a_n, a_phys, tr, tt = batch
            s0_n = s0_n.to(DEVICE); s1_n = s1_n.to(DEVICE); a_n = a_n.to(DEVICE)
            pred_next_n, _ = predict_next_norm(
                model, model_type, pde_name, s0_n, a_n, normalizer,
                x_norm_1d=x_norm_1d.to(DEVICE) if x_norm_1d is not None else None,
                coords_2d=coords_2d.to(DEVICE) if coords_2d is not None else None,
                coord_feat_1d=coord_feat_1d.to(DEVICE) if coord_feat_1d is not None else None,
                coord_feat_2d=coord_feat_2d.to(DEVICE) if coord_feat_2d is not None else None,
                residual_scale_np=residual_scale_np,
            )
            pred_phys = normalizer.denorm_state_t(pred_next_n)
            true_phys = normalizer.denorm_state_t(s1_n)
            diff = pred_phys - true_phys
            B = diff.shape[0]
            rel = torch.norm(diff.reshape(B, -1), dim=1) / (torch.norm(true_phys.reshape(B, -1), dim=1) + 1e-12)
            mae = torch.mean(torch.abs(diff).reshape(B, -1), dim=1)
            rel_l2_list.append(rel.cpu().numpy())
            mae_list.append(mae.cpu().numpy())
            if n_risk < max_risk_samples:
                pred_np = pred_phys.detach().cpu().numpy()
                true_np = true_phys.detach().cpu().numpy()
                take = min(B, max_risk_samples - n_risk)
                for i in range(take):
                    risk_pred.append(risk_scalar_np(pred_np[i], pde_name, meta))
                    risk_true.append(risk_scalar_np(true_np[i], pde_name, meta))
                n_risk += take
    rel_all = np.concatenate(rel_l2_list)
    mae_all = np.concatenate(mae_list)
    if len(risk_pred) >= 3:
        rp = np.asarray(risk_pred)
        rt = np.asarray(risk_true)
        pear = float(np.corrcoef(rp, rt)[0, 1]) if np.std(rp) > 1e-12 and np.std(rt) > 1e-12 else float("nan")
        # Spearman without scipy
        rank_p = np.argsort(np.argsort(rp))
        rank_t = np.argsort(np.argsort(rt))
        spear = float(np.corrcoef(rank_p, rank_t)[0, 1]) if np.std(rank_p) > 1e-12 and np.std(rank_t) > 1e-12 else float("nan")
    else:
        pear, spear = float("nan"), float("nan")
    return {
        "rel_l2_mean": float(rel_all.mean()),
        "rel_l2_std": float(rel_all.std()),
        "rel_l2_q95": float(np.quantile(rel_all, 0.95)),
        "mae_mean": float(mae_all.mean()),
        "mae_std": float(mae_all.std()),
        "risk_pearson": pear,
        "risk_spearman": spear,
        "rel_l2_errors": rel_all,
        "mae_errors": mae_all,
    }


# ============================================================
# Plot helpers
# ============================================================
def plot_training_curves(history: Dict[str, List[float]], save_dir: str, title: str):
    plt.figure(figsize=(7.2, 4.4))
    for key in ["train_total", "val_total", "train_data", "val_data", "train_inc", "train_roll", "train_risk_roll", "train_grad_roll", "train_grad", "train_phys", "train_bound", "val_rollout_raw", "val_stable_score"]:
        if key in history:
            plt.plot(history[key], label=key)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=CONFIG["dpi"])
    plt.close()


def plot_hist(x: np.ndarray, save_dir: str, fname: str, title: str, xlabel: str):
    plt.figure(figsize=(6.0, 4.0))
    plt.hist(x, bins=40, alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=CONFIG["dpi"])
    plt.close()


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def get_rollout_curriculum(epoch: int, pde_name: str) -> Tuple[float, int]:
    warm = int(CONFIG.get("rollout_warmup_epochs", 0))
    if epoch < warm:
        return 0.0, 0
    base_w = float(CONFIG.get("rollout_loss_weight", 0.0))
    ramp_w = max(1, int(CONFIG.get("rollout_loss_ramp_epochs", 1)))
    prog_w = min(1.0, float(epoch - warm + 1) / float(ramp_w))
    roll_w = base_w * prog_w

    k_start = int(CONFIG.get("rollout_k_start", 1))
    k_cfg = CONFIG.get("rollout_k_max", k_start)
    if isinstance(k_cfg, dict):
        k_max = int(k_cfg.get(pde_name, k_start))
    else:
        k_max = int(k_cfg)
    ramp_k = max(1, int(CONFIG.get("rollout_k_ramp_epochs", 1)))
    prog_k = min(1.0, float(epoch - warm + 1) / float(ramp_k))
    roll_k = int(round(k_start + (k_max - k_start) * prog_k))
    roll_k = max(k_start, min(k_max, roll_k)) if roll_w > 0 else 0
    return roll_w, roll_k


def evaluate_rollout_subset(model: nn.Module, model_type: str, pde_name: str,
                            state: np.ndarray, control: np.ndarray, traj_ids: np.ndarray,
                            rollout_steps: int, normalizer: Normalizer, meta: dict,
                            x_norm_1d=None, coords_2d=None, coord_feat_1d=None, coord_feat_2d=None,
                            residual_scale_np: Optional[np.ndarray] = None,
                            projected_eval: bool = True) -> Dict[str, float]:
    raw_rel, raw_rmse = [], []
    proj_rel, proj_rmse = [], []
    raw_risk_abs, proj_risk_abs = [], []
    proj_diag_all: Dict[str, List[float]] = {
        "clip_frac_mean": [], "clip_frac_max": [],
        "upper_saturation_frac_mean": [], "upper_saturation_frac_max": [],
        "lower_saturation_frac_mean": [], "lower_saturation_frac_max": [],
        "projection_rel_l2_mean": [], "projection_rel_l2_max": [],
    }
    for tr in [int(x) for x in np.asarray(traj_ids).tolist()]:
        controls_phys = control[tr, :rollout_steps]
        if pde_name == "burgers":
            s0_phys = state[tr, 0][None, :]
            truth = state[tr, :rollout_steps + 1][:, None, :]
        else:
            s0_phys = state[tr, 0]
            truth = state[tr, :rollout_steps + 1]

        pred_raw, _ = rollout_model_np(
            model, model_type, pde_name, s0_phys, controls_phys, normalizer,
            x_norm_1d, coords_2d, coord_feat_1d, coord_feat_2d,
            residual_scale_np=residual_scale_np,
            projected=False,
            return_diag=True,
        )
        Lraw = min(len(pred_raw), len(truth))
        diff_raw = pred_raw[Lraw - 1] - truth[Lraw - 1]
        raw_rel.append(float(np.linalg.norm(diff_raw.reshape(-1)) / (np.linalg.norm(truth[Lraw - 1].reshape(-1)) + 1e-12)))
        raw_rmse.append(float(np.sqrt(np.mean(diff_raw ** 2))))
        try:
            raw_risk_abs.append(abs(risk_scalar_np(pred_raw[Lraw - 1], pde_name, meta) - risk_scalar_np(truth[Lraw - 1], pde_name, meta)))
        except Exception:
            pass

        if projected_eval:
            pred_proj, diag_proj = rollout_model_np(
                model, model_type, pde_name, s0_phys, controls_phys, normalizer,
                x_norm_1d, coords_2d, coord_feat_1d, coord_feat_2d,
                residual_scale_np=residual_scale_np,
                projected=True,
                return_diag=True,
            )
            Lproj = min(len(pred_proj), len(truth))
            diff_proj = pred_proj[Lproj - 1] - truth[Lproj - 1]
            proj_rel.append(float(np.linalg.norm(diff_proj.reshape(-1)) / (np.linalg.norm(truth[Lproj - 1].reshape(-1)) + 1e-12)))
            proj_rmse.append(float(np.sqrt(np.mean(diff_proj ** 2))))
            try:
                proj_risk_abs.append(abs(risk_scalar_np(pred_proj[Lproj - 1], pde_name, meta) - risk_scalar_np(truth[Lproj - 1], pde_name, meta)))
            except Exception:
                pass
            for k in proj_diag_all:
                proj_diag_all[k].append(float(diag_proj.get(k, 0.0)))

    out = {
        "rollout_rel_final_raw": float(np.mean(raw_rel)) if raw_rel else float("nan"),
        "rollout_rmse_final_raw": float(np.mean(raw_rmse)) if raw_rmse else float("nan"),
        "rollout_rel_final_projected": float(np.mean(proj_rel)) if proj_rel else float("nan"),
        "rollout_rmse_final_projected": float(np.mean(proj_rmse)) if proj_rmse else float("nan"),
        "rollout_risk_abs_final_raw": float(np.mean(raw_risk_abs)) if raw_risk_abs else float("nan"),
        "rollout_risk_abs_final_projected": float(np.mean(proj_risk_abs)) if proj_risk_abs else float("nan"),
    }
    out.update({k: float(np.mean(v)) if len(v) else 0.0 for k, v in proj_diag_all.items()})
    return out


def evaluate_rollout_multihorizon_subset(model: nn.Module, model_type: str, pde_name: str,
                                         state: np.ndarray, control: np.ndarray, traj_ids: np.ndarray,
                                         horizons: List[int], normalizer: Normalizer, meta: dict,
                                         x_norm_1d=None, coords_2d=None, coord_feat_1d=None, coord_feat_2d=None,
                                         residual_scale_np: Optional[np.ndarray] = None,
                                         projected_eval: bool = True) -> Dict[str, float]:
    """Evaluate rollout at several horizons and return weighted aggregate fields."""
    weights_cfg = CONFIG.get("rollout_select_weights_horizon", {})
    max_h = control.shape[1]
    hs = [int(h) for h in horizons if int(h) > 0 and int(h) <= max_h]
    if not hs:
        hs = [min(max_h, int(CONFIG.get("rollout_steps_2d", 80)))]
    agg: Dict[str, float] = {}
    wsum = 0.0
    raw_score = 0.0
    risk_score = 0.0
    for h in hs:
        d = evaluate_rollout_subset(model, model_type, pde_name, state, control, traj_ids, h, normalizer, meta,
                                    x_norm_1d=x_norm_1d, coords_2d=coords_2d,
                                    coord_feat_1d=coord_feat_1d, coord_feat_2d=coord_feat_2d,
                                    residual_scale_np=residual_scale_np, projected_eval=projected_eval)
        for k, v in d.items():
            agg[f"H{h}_{k}"] = float(v)
        w = float(weights_cfg.get(h, weights_cfg.get(str(h), 1.0)))
        wsum += w
        raw_score += w * float(d.get("rollout_rel_final_raw", float("inf")))
        risk_score += w * float(d.get("rollout_risk_abs_final_raw", 0.0))
    agg["rollout_rel_final_raw"] = raw_score / max(wsum, 1.0e-12)
    agg["rollout_risk_abs_final_raw"] = risk_score / max(wsum, 1.0e-12)
    # For plotting/backward compatibility, use the largest horizon's projected stats.
    hlast = hs[-1]
    for key in ["rollout_rel_final_projected", "rollout_rmse_final_projected", "clip_frac_mean",
                "lower_saturation_frac_mean", "upper_saturation_frac_mean", "projection_rel_l2_mean"]:
        agg[key] = agg.get(f"H{hlast}_{key}", float("nan"))
    return agg


def compute_stable_selection_score(val_loss: float, best_val: float, rollout_diag: Dict[str, float]) -> float:
    w = CONFIG.get("stable_select_weights", {"val": 0.2, "raw_rollout": 1.0, "lower_sat": 0.6, "clip_frac": 0.2})
    val_norm = float(val_loss / max(best_val, 1.0e-12))
    raw_roll = float(rollout_diag.get("rollout_rel_final_raw", float("inf")))
    risk_err = float(rollout_diag.get("rollout_risk_abs_final_raw", 0.0))
    lower_sat = float(rollout_diag.get("lower_saturation_frac_mean", 0.0))
    clip_frac = float(rollout_diag.get("clip_frac_mean", 0.0))
    return float(w.get("raw_rollout", 1.0) * raw_roll
                 + w.get("risk", 0.0) * risk_err
                 + w.get("val", 0.2) * val_norm
                 + w.get("lower_sat", 0.6) * lower_sat
                 + w.get("clip_frac", 0.2) * clip_frac)


# ============================================================
# Training one model
# ============================================================
def prepare_coords_and_features(pde_name: str, data_bundle: dict):
    meta = data_bundle["meta"]
    if pde_name == "burgers":
        x = data_bundle["grid_x"]
        x_norm_np = (2.0 * (x / meta["L"]) - 1.0).astype(np.float32)
        x_norm = torch.from_numpy(x_norm_np)
        coord_feat_1d = fourier_features_1d(x_norm, CONFIG["fourier_freqs"]).float()
        return x_norm, None, coord_feat_1d, None, coord_feat_1d.shape[-1]
    else:
        x = data_bundle["grid_x"]
        y = data_bundle["grid_y"]
        xg = 2.0 * (x / meta["Lx"]) - 1.0
        yg = 2.0 * (y / meta["Ly"]) - 1.0
        Xn, Yn = np.meshgrid(xg, yg)
        coords_np = np.stack([Xn, Yn], axis=-1).astype(np.float32)
        coords_2d = torch.from_numpy(coords_np)
        coord_feat_2d = fourier_features_2d(coords_2d, CONFIG["fourier_freqs"]).float()
        return None, coords_2d, None, coord_feat_2d, coord_feat_2d.shape[-1]


def train_one_model(model_type: str, pde_name: str, data_bundle: dict,
                    train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray,
                    result_root: str):
    save_dir = os.path.join(result_root, pde_name, model_type)
    ensure_dir(save_dir)
    state = data_bundle["state"]
    control = data_bundle["control"]
    meta = data_bundle["meta"]
    print(f"\n{'='*78}\nTraining {model_type.upper()} for controlled {pde_name.upper()}\n{'='*78}")

    normalizer = Normalizer.fit(state, control, train_ids)
    save_json(asdict(normalizer.stats), os.path.join(save_dir, "normalizer_stats.json"))

    residual_scale_np = compute_residual_scale_np(
        state, train_ids, normalizer,
        max_pairs=int(CONFIG.get("residual_scale_max_pairs", 20000)),
        quantile=float(CONFIG.get("residual_limit_quantile", 0.995)),
        multiplier=float(CONFIG.get("residual_limit_multiplier", 2.0)),
        seed=int(CONFIG["seed"]),
    )
    save_json({
        "residual_scale_norm": residual_scale_np.astype(float).tolist(),
        "quantile": float(CONFIG.get("residual_limit_quantile", 0.995)),
        "multiplier": float(CONFIG.get("residual_limit_multiplier", 2.0)),
        "residual_step_limit": bool(CONFIG.get("residual_step_limit", True)),
    }, os.path.join(save_dir, "residual_step_scale.json"))

    train_ds = NormalizedPairDataset(state, control, train_ids, normalizer, pde_name)
    val_ds = NormalizedPairDataset(state, control, val_ids, normalizer, pde_name)
    test_ds = NormalizedPairDataset(state, control, test_ids, normalizer, pde_name)

    if pde_name == "burgers":
        batch_size = CONFIG["operator_batch_size_1d"] if model_type in ["fno", "pino"] else CONFIG["coord_batch_size_1d"]
        state_shape = (1, state.shape[-1])
        rollout_steps = min(CONFIG["rollout_steps_1d"], meta["nt"])
    else:
        batch_size = CONFIG["operator_batch_size_2d"] if model_type in ["fno", "pino"] else CONFIG["coord_batch_size_2d"]
        state_shape = (state.shape[2], state.shape[-2], state.shape[-1])
        rollout_steps = min(CONFIG["rollout_steps_2d"], meta["nt"])

    num_workers = int(CONFIG.get("num_workers", 0))
    loader_kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    x_norm_1d, coords_2d, coord_feat_1d, coord_feat_2d, coord_dim = prepare_coords_and_features(pde_name, data_bundle)
    model = get_model(model_type, pde_name, state_shape, coord_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, CONFIG["epochs"]), eta_min=CONFIG["lr"] * 0.05)
    # IMPORTANT AMP policy:
    # FNO/PINO use complex-valued spectral weights. PyTorch GradScaler cannot
    # unscale complex CUDA gradients, so AMP is automatically disabled for
    # models with complex parameters while staying enabled for PINN/DeepONet.
    has_complex_params = any(torch.is_complex(p) for p in model.parameters())
    amp_enabled = bool(CONFIG["amp"] and torch.cuda.is_available() and (not has_complex_params))
    if bool(CONFIG["amp"] and torch.cuda.is_available()) and has_complex_params:
        print(f"[Info] AMP disabled for {model_type}+{pde_name} because the model has complex spectral parameters.")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    actuator_t = torch.from_numpy(data_bundle["actuator"].astype(np.float32)).to(DEVICE)
    if pde_name == "kolmogorov":
        x = data_bundle["grid_x"]
        y = data_bundle["grid_y"]
        X, Y = np.meshgrid(x, y)
        forcing = meta["forcing_amp"] * meta["forcing_k"] * np.cos(meta["forcing_k"] * Y)
        forcing_t = torch.from_numpy(forcing.astype(np.float32)).to(DEVICE)
    else:
        forcing_t = None

    history: Dict[str, List[float]] = {k: [] for k in [
        "train_total", "train_data", "train_inc", "train_roll", "train_risk_roll", "train_grad_roll", "train_grad", "train_phys", "train_bound",
        "val_total", "val_data", "val_rollout_raw", "val_rollout_projected",
        "val_lower_sat", "val_clip_frac", "val_stable_score"
    ]}
    best_val = float("inf")
    best_val_state = None
    best_val_epoch = -1
    best_rollout_raw = float("inf")
    best_rollout_state = None
    best_rollout_epoch = -1
    best_stable_score = float("inf")
    best_stable_state = None
    best_stable_epoch = -1
    best_rollout_diag: Dict[str, float] = {}
    best_stable_diag: Dict[str, float] = {}
    val_roll_ids = np.asarray(val_ids[:min(int(CONFIG.get("num_val_rollout_eval_traj", 6)), len(val_ids))], dtype=int)

    pbar = tqdm(range(CONFIG["epochs"]), desc=f"train {model_type}+{pde_name}")
    for ep in pbar:
        model.train()
        sums = {"total": 0.0, "data": 0.0, "inc": 0.0, "roll": 0.0, "risk_roll": 0.0, "grad_roll": 0.0, "grad": 0.0, "phys": 0.0, "bound": 0.0}
        nb = 0
        phys_base_w = get_base_physics_weight(model_type, pde_name, ep)
        grad_w = CONFIG["gradient_loss_weight"].get(pde_name, 0.0)
        roll_w, roll_k = get_rollout_curriculum(ep, pde_name)
        bound_w_lower, bound_w_upper = get_bound_weights(pde_name, ep)

        for batch_i, batch in enumerate(train_loader):
            s0_n, delta_n, s1_n, a_n, a_phys, tr, tt = batch
            s0_n = s0_n.to(DEVICE, non_blocking=True)
            delta_n = delta_n.to(DEVICE, non_blocking=True)
            s1_n = s1_n.to(DEVICE, non_blocking=True)
            a_n = a_n.to(DEVICE, non_blocking=True)
            a_phys = a_phys.to(DEVICE, non_blocking=True)
            noise_std = float(CONFIG.get("input_noise_std_norm", 0.0))
            if noise_std > 0.0 and ep >= int(CONFIG.get("input_noise_warmup_epochs", 0)):
                s0_n = s0_n + noise_std * torch.randn_like(s0_n)
            tr = tr.to(DEVICE); tt = tt.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred_next_n, delta_pred, delta_raw_pred = predict_next_norm_with_raw(
                    model, model_type, pde_name, s0_n, a_n, normalizer,
                    x_norm_1d=x_norm_1d.to(DEVICE) if x_norm_1d is not None else None,
                    coords_2d=coords_2d.to(DEVICE) if coords_2d is not None else None,
                    coord_feat_1d=coord_feat_1d.to(DEVICE) if coord_feat_1d is not None else None,
                    coord_feat_2d=coord_feat_2d.to(DEVICE) if coord_feat_2d is not None else None,
                    residual_scale_np=residual_scale_np,
                )
                data_l = F.mse_loss(pred_next_n, s1_n)
                inc_w = get_pde_model_weight("increment_loss_weight", pde_name, model_type, 0.0)
                inc_l = increment_fit_loss(delta_raw_pred, delta_pred, delta_n, residual_scale_np) if inc_w > 0 else torch.tensor(0.0, device=DEVICE)
                pred_phys = normalizer.denorm_state_t(pred_next_n.float())
                target_phys = normalizer.denorm_state_t(s1_n.float())
                curr_phys = normalizer.denorm_state_t(s0_n.float())
                if (bound_w_lower > 0) or (bound_w_upper > 0):
                    bound_low_l, bound_high_l = physical_bound_components(pred_phys, pde_name)
                    bound_l = bound_w_lower * bound_low_l + bound_w_upper * bound_high_l
                else:
                    bound_low_l = torch.tensor(0.0, device=DEVICE)
                    bound_high_l = torch.tensor(0.0, device=DEVICE)
                    bound_l = torch.tensor(0.0, device=DEVICE)
                grad_l = gradient_risk_loss(pred_phys, target_phys, pde_name, meta) if grad_w > 0 else torch.tensor(0.0, device=DEVICE)
                if phys_base_w > 0:
                    phys_l = physics_loss_dispatch(pde_name, pred_phys, curr_phys, a_phys, actuator_t, meta, forcing_t)
                    if CONFIG["physics_dynamic_balance"]:
                        balance = (data_l.detach() / (phys_l.detach() + 1e-12)).clamp(max=CONFIG["physics_balance_clip"])
                        phys_eff_w = phys_base_w * balance
                    else:
                        phys_eff_w = phys_base_w
                else:
                    phys_l = torch.tensor(0.0, device=DEVICE)
                    phys_eff_w = 0.0
                do_train_roll = (roll_w > 0) and ((batch_i % max(1, int(CONFIG.get("rollout_train_every", 1)))) == 0)
                if do_train_roll:
                    max_rb = int(CONFIG.get("rollout_train_max_batch", 0))
                    if max_rb > 0 and s0_n.shape[0] > max_rb:
                        sel = torch.randperm(s0_n.shape[0], device=s0_n.device)[:max_rb]
                        s0_roll, tr_roll, tt_roll = s0_n[sel], tr[sel], tt[sel]
                    else:
                        s0_roll, tr_roll, tt_roll = s0_n, tr, tt
                    roll_l, roll_bound_l, risk_roll_l, grad_roll_l = compute_rollout_loss(
                        model, model_type, pde_name, train_ds, tr_roll, tt_roll, s0_roll,
                        normalizer,
                        x_norm_1d.to(DEVICE) if x_norm_1d is not None else None,
                        coords_2d.to(DEVICE) if coords_2d is not None else None,
                        coord_feat_1d.to(DEVICE) if coord_feat_1d is not None else None,
                        coord_feat_2d.to(DEVICE) if coord_feat_2d is not None else None,
                        roll_k, residual_scale_np=residual_scale_np, meta=meta, epoch=ep)
                else:
                    roll_l = torch.tensor(0.0, device=DEVICE)
                    roll_bound_l = torch.tensor(0.0, device=DEVICE)
                    risk_roll_l = torch.tensor(0.0, device=DEVICE)
                    grad_roll_l = torch.tensor(0.0, device=DEVICE)
                risk_roll_w = get_pde_model_weight("risk_rollout_weight", pde_name, model_type, 0.0)
                grad_roll_w = get_pde_model_weight("grad_rollout_weight", pde_name, model_type, 0.0)
                loss = (data_l + inc_w * inc_l + grad_w * grad_l + phys_eff_w * phys_l
                        + roll_w * roll_l
                        + roll_w * risk_roll_w * risk_roll_l
                        + roll_w * grad_roll_w * grad_roll_l
                        + bound_l + 0.5 * (bound_w_lower + bound_w_upper) * roll_bound_l)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            sums["total"] += float(loss.detach().cpu())
            sums["data"] += float(data_l.detach().cpu())
            sums["inc"] += float(inc_l.detach().cpu())
            sums["roll"] += float(roll_l.detach().cpu())
            sums["risk_roll"] += float(risk_roll_l.detach().cpu())
            sums["grad_roll"] += float(grad_roll_l.detach().cpu())
            sums["grad"] += float(grad_l.detach().cpu())
            sums["phys"] += float(phys_l.detach().cpu())
            sums["bound"] += float(bound_l.detach().cpu())
            nb += 1

        scheduler.step()
        for k in sums:
            history[f"train_{k}"].append(sums[k] / max(nb, 1))

        # validation: one-step normalized residual loss only + gradient optional
        model.eval()
        vt, vd, vb = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                s0_n, delta_n, s1_n, a_n, a_phys, tr, tt = batch
                s0_n = s0_n.to(DEVICE); delta_n = delta_n.to(DEVICE); a_n = a_n.to(DEVICE)
                pred_next_n, _ = predict_next_norm(
                    model, model_type, pde_name, s0_n, a_n, normalizer,
                    x_norm_1d=x_norm_1d.to(DEVICE) if x_norm_1d is not None else None,
                    coords_2d=coords_2d.to(DEVICE) if coords_2d is not None else None,
                    coord_feat_1d=coord_feat_1d.to(DEVICE) if coord_feat_1d is not None else None,
                    coord_feat_2d=coord_feat_2d.to(DEVICE) if coord_feat_2d is not None else None,
                    residual_scale_np=residual_scale_np,
                )
                dl = F.mse_loss(pred_next_n, s1_n.to(DEVICE))
                vt += float(dl.detach().cpu())
                vd += float(dl.detach().cpu())
                vb += 1
        val_loss = vt / max(vb, 1)
        history["val_total"].append(val_loss)
        history["val_data"].append(vd / max(vb, 1))

        if val_loss < best_val:
            best_val = val_loss
            best_val_epoch = ep
            best_val_state = clone_state_dict(model)

        do_rollout_val = (
            len(val_roll_ids) > 0
            and ep >= int(CONFIG.get("selection_rollout_warmup_epochs", CONFIG.get("rollout_warmup_epochs", 0)))
            and (((ep + 1) % int(CONFIG.get("rollout_val_eval_period", 5))) == 0 or ep == CONFIG["epochs"] - 1)
        )
        if do_rollout_val:
            val_roll_diag = evaluate_rollout_multihorizon_subset(
                model, model_type, pde_name,
                state, control, val_roll_ids,
                [int(h) for h in CONFIG.get("rollout_select_horizons", [rollout_steps])], normalizer, meta,
                x_norm_1d=x_norm_1d, coords_2d=coords_2d,
                coord_feat_1d=coord_feat_1d, coord_feat_2d=coord_feat_2d,
                residual_scale_np=residual_scale_np,
                projected_eval=bool(CONFIG.get("evaluate_projected_rollout", True)),
            )
            history["val_rollout_raw"].append(float(val_roll_diag.get("rollout_rel_final_raw", float("nan"))))
            history["val_rollout_projected"].append(float(val_roll_diag.get("rollout_rel_final_projected", float("nan"))))
            history["val_lower_sat"].append(float(val_roll_diag.get("lower_saturation_frac_mean", float("nan"))))
            history["val_clip_frac"].append(float(val_roll_diag.get("clip_frac_mean", float("nan"))))
            stable_score = compute_stable_selection_score(val_loss, best_val, val_roll_diag)
            history["val_stable_score"].append(stable_score)

            eligible = val_loss <= float(CONFIG.get("selection_val_guard_ratio", 1.35)) * best_val
            if eligible and float(val_roll_diag.get("rollout_rel_final_raw", float("inf"))) < best_rollout_raw:
                best_rollout_raw = float(val_roll_diag.get("rollout_rel_final_raw", float("inf")))
                best_rollout_epoch = ep
                best_rollout_state = clone_state_dict(model)
                best_rollout_diag = dict(val_roll_diag)
            if eligible and stable_score < best_stable_score:
                best_stable_score = stable_score
                best_stable_epoch = ep
                best_stable_state = clone_state_dict(model)
                best_stable_diag = dict(val_roll_diag)
        else:
            history["val_rollout_raw"].append(float("nan"))
            history["val_rollout_projected"].append(float("nan"))
            history["val_lower_sat"].append(float("nan"))
            history["val_clip_frac"].append(float("nan"))
            history["val_stable_score"].append(float("nan"))

        postfix = {"train": history["train_total"][-1], "val": val_loss, "roll_w": roll_w, "roll_k": roll_k, "phys_w": phys_base_w}
        if do_rollout_val:
            postfix["vroll"] = history["val_rollout_raw"][-1]
        pbar.set_postfix(**postfix)

    if best_val_state is None:
        raise RuntimeError("No best checkpoint was saved; training failed before first validation.")

    checkpoint_selection = str(CONFIG.get("default_checkpoint_selection", "best_stable"))
    selected_state = None
    selected_epoch = -1
    selected_tag = checkpoint_selection
    selected_rollout_diag: Dict[str, float] = {}
    if checkpoint_selection == "best_rollout" and best_rollout_state is not None:
        selected_state = best_rollout_state
        selected_epoch = best_rollout_epoch
        selected_rollout_diag = dict(best_rollout_diag)
    elif checkpoint_selection == "best_stable" and best_stable_state is not None:
        selected_state = best_stable_state
        selected_epoch = best_stable_epoch
        selected_rollout_diag = dict(best_stable_diag)
    else:
        selected_state = best_val_state
        selected_epoch = best_val_epoch
        selected_tag = "best_val"

    model.load_state_dict(selected_state)
    model.eval()

    # Evaluation
    eval_metrics = eval_one_step_with_risk(model, model_type, pde_name, test_loader, normalizer, meta,
                                           x_norm_1d=x_norm_1d, coords_2d=coords_2d,
                                           coord_feat_1d=coord_feat_1d, coord_feat_2d=coord_feat_2d,
                                           residual_scale_np=residual_scale_np)
    rel_l2_errors = eval_metrics.pop("rel_l2_errors")
    mae_errors = eval_metrics.pop("mae_errors")

    # Rollout eval on several test trajectories
    rollout_rel_finals_raw = []
    rollout_rmse_finals_raw = []
    rollout_rel_finals_projected = []
    rollout_rmse_finals_projected = []
    proj_diag_all: Dict[str, List[float]] = {
        "clip_frac_mean": [], "clip_frac_max": [],
        "upper_saturation_frac_mean": [], "upper_saturation_frac_max": [],
        "lower_saturation_frac_mean": [], "lower_saturation_frac_max": [],
        "projection_rel_l2_mean": [], "projection_rel_l2_max": [],
    }

    viz_rollout_truth = None
    viz_rollout_pred_raw = None
    viz_rollout_pred_projected = None
    viz_rollout_ctrl = None

    n_roll_eval = min(CONFIG["num_rollout_eval_traj"], len(test_ids))
    for jj in range(n_roll_eval):
        tr = int(test_ids[jj])
        controls_phys = control[tr, :rollout_steps]
        if pde_name == "burgers":
            s0_phys = state[tr, 0][None, :]
            truth = state[tr, :rollout_steps + 1]
            truth = truth[:, None, :]
        else:
            s0_phys = state[tr, 0]
            truth = state[tr, :rollout_steps + 1]

        pred_raw, _ = rollout_model_np(
            model, model_type, pde_name, s0_phys, controls_phys, normalizer,
            x_norm_1d, coords_2d, coord_feat_1d, coord_feat_2d,
            residual_scale_np=residual_scale_np,
            projected=False,
            return_diag=True,
        )
        Lraw = min(len(pred_raw), len(truth))
        diff_final_raw = pred_raw[Lraw - 1] - truth[Lraw - 1]
        rollout_rel_finals_raw.append(float(np.linalg.norm(diff_final_raw.reshape(-1)) / (np.linalg.norm(truth[Lraw - 1].reshape(-1)) + 1e-12)))
        rollout_rmse_finals_raw.append(float(np.sqrt(np.mean(diff_final_raw ** 2))))

        if CONFIG.get("evaluate_projected_rollout", True):
            pred_proj, diag_proj = rollout_model_np(
                model, model_type, pde_name, s0_phys, controls_phys, normalizer,
                x_norm_1d, coords_2d, coord_feat_1d, coord_feat_2d,
                residual_scale_np=residual_scale_np,
                projected=True,
                return_diag=True,
            )
            Lproj = min(len(pred_proj), len(truth))
            diff_final_proj = pred_proj[Lproj - 1] - truth[Lproj - 1]
            rollout_rel_finals_projected.append(float(np.linalg.norm(diff_final_proj.reshape(-1)) / (np.linalg.norm(truth[Lproj - 1].reshape(-1)) + 1e-12)))
            rollout_rmse_finals_projected.append(float(np.sqrt(np.mean(diff_final_proj ** 2))))
            for k in proj_diag_all:
                proj_diag_all[k].append(float(diag_proj.get(k, 0.0)))
        else:
            pred_proj = np.array([], dtype=np.float32)

        if jj == 0:
            viz_rollout_truth = truth.astype(np.float32)
            viz_rollout_pred_raw = pred_raw.astype(np.float32)
            viz_rollout_pred_projected = pred_proj.astype(np.float32) if pred_proj.size else np.array([])
            viz_rollout_ctrl = controls_phys.astype(np.float32)

    rollout_rel_final_raw = float(np.mean(rollout_rel_finals_raw)) if rollout_rel_finals_raw else float("nan")
    rollout_rmse_final_raw = float(np.mean(rollout_rmse_finals_raw)) if rollout_rmse_finals_raw else float("nan")
    rollout_rel_final_projected = float(np.mean(rollout_rel_finals_projected)) if rollout_rel_finals_projected else float("nan")
    rollout_rmse_final_projected = float(np.mean(rollout_rmse_finals_projected)) if rollout_rmse_finals_projected else float("nan")
    proj_summary = {k: float(np.mean(v)) if len(v) else 0.0 for k, v in proj_diag_all.items()}

    # Explicit final test evaluation at every reported rollout horizon.
    test_multi_horizon_ids = np.asarray(test_ids[:min(int(CONFIG.get("multi_horizon_max_traj_per_eval", 16)), len(test_ids))], dtype=int)
    test_multi_horizon_diag = evaluate_rollout_multihorizon_subset(
        model, model_type, pde_name,
        state, control, test_multi_horizon_ids,
        [int(h) for h in CONFIG.get("rollout_select_horizons", [rollout_steps])],
        normalizer, meta,
        x_norm_1d=x_norm_1d, coords_2d=coords_2d,
        coord_feat_1d=coord_feat_1d, coord_feat_2d=coord_feat_2d,
        residual_scale_np=residual_scale_np,
        projected_eval=bool(CONFIG.get("evaluate_projected_rollout", True)),
    )

    # Backward-compatible fields keep the honest raw rollout, not the projected diagnostic.
    rollout_rel_final = rollout_rel_final_raw
    rollout_rmse_final = rollout_rmse_final_raw

    # checkpoint
    ckpt = {
        "state_dict": model.state_dict(),
        "model_type": model_type,
        "pde_name": pde_name,
        "architecture_version": "risk_aware_long_rollout",
        "normalizer": asdict(normalizer.stats),
        "residual_scale_norm": residual_scale_np.astype(float).tolist(),
        "stability_config": {
            "residual_step_limit": bool(CONFIG.get("residual_step_limit", True)),
            "residual_limit_quantile": float(CONFIG.get("residual_limit_quantile", 0.995)),
            "residual_limit_multiplier": float(CONFIG.get("residual_limit_multiplier", 2.0)),
            "bound_loss_weight_lower": CONFIG.get("bound_loss_weight_lower", {}),
            "bound_loss_weight_upper": CONFIG.get("bound_loss_weight_upper", {}),
            "bound_warmup_epochs": int(CONFIG.get("bound_warmup_epochs", 0)),
        },
        "cfg": {
            "experiment": {"mode": model_type, "controlled": True, "pde": pde_name, "seed": CONFIG["seed"]},
            "training": {"epochs": CONFIG["epochs"], "lr": CONFIG["lr"], "weight_decay": CONFIG["weight_decay"], "residual_learning": True},
            "model_config": {
                "state_shape": state_shape,
                "coord_dim": coord_dim,
                "fourier_freqs": CONFIG["fourier_freqs"],
            },
            "data_meta": meta,
            "split_info": {"train_ids": train_ids.tolist(), "val_ids": val_ids.tolist(), "test_ids": test_ids.tolist()},
        },
        "best_val_loss": float(best_val),
        "best_val_epoch": int(best_val_epoch),
        "best_rollout_raw": float(best_rollout_raw) if np.isfinite(best_rollout_raw) else None,
        "best_rollout_epoch": int(best_rollout_epoch),
        "best_stable_score": float(best_stable_score) if np.isfinite(best_stable_score) else None,
        "best_stable_epoch": int(best_stable_epoch),
        "selected_checkpoint_tag": selected_tag,
        "selected_checkpoint_epoch": int(selected_epoch),
        "selected_rollout_diag": selected_rollout_diag,
    }
    torch.save(ckpt, os.path.join(save_dir, "best_checkpoint.pt"))
    if best_val_state is not None:
        ckpt_best_val = dict(ckpt)
        ckpt_best_val["state_dict"] = best_val_state
        ckpt_best_val["selected_checkpoint_tag"] = "best_val"
        ckpt_best_val["selected_checkpoint_epoch"] = int(best_val_epoch)
        torch.save(ckpt_best_val, os.path.join(save_dir, "best_validation_checkpoint.pt"))
    if best_rollout_state is not None:
        ckpt_best_roll = dict(ckpt)
        ckpt_best_roll["state_dict"] = best_rollout_state
        ckpt_best_roll["selected_checkpoint_tag"] = "best_rollout"
        ckpt_best_roll["selected_checkpoint_epoch"] = int(best_rollout_epoch)
        ckpt_best_roll["selected_rollout_diag"] = best_rollout_diag
        torch.save(ckpt_best_roll, os.path.join(save_dir, "best_rollout_checkpoint.pt"))
    if best_stable_state is not None:
        ckpt_best_stable = dict(ckpt)
        ckpt_best_stable["state_dict"] = best_stable_state
        ckpt_best_stable["selected_checkpoint_tag"] = "best_stable"
        ckpt_best_stable["selected_checkpoint_epoch"] = int(best_stable_epoch)
        ckpt_best_stable["selected_rollout_diag"] = best_stable_diag
        torch.save(ckpt_best_stable, os.path.join(save_dir, "best_stability_checkpoint.pt"))

    # plots/results
    plot_training_curves(history, save_dir, f"{model_type.upper()} + controlled {pde_name.upper()}")
    plot_hist(rel_l2_errors, save_dir, "test_relative_l2_histogram.png", f"{model_type.upper()} + {pde_name.upper()} test relative L2", "relative L2 error")

    np.savez_compressed(
        os.path.join(save_dir, "results.npz"),
        rel_l2_errors=rel_l2_errors.astype(np.float32),
        mae_errors=mae_errors.astype(np.float32),
        train_total=np.asarray(history["train_total"], dtype=np.float32),
        val_total=np.asarray(history["val_total"], dtype=np.float32),
        train_data=np.asarray(history["train_data"], dtype=np.float32),
        train_inc=np.asarray(history["train_inc"], dtype=np.float32),
        train_roll=np.asarray(history["train_roll"], dtype=np.float32),
        train_risk_roll=np.asarray(history["train_risk_roll"], dtype=np.float32),
        train_grad_roll=np.asarray(history["train_grad_roll"], dtype=np.float32),
        train_grad=np.asarray(history["train_grad"], dtype=np.float32),
        train_phys=np.asarray(history["train_phys"], dtype=np.float32),
        train_bound=np.asarray(history["train_bound"], dtype=np.float32),
        val_rollout_raw=np.asarray(history["val_rollout_raw"], dtype=np.float32),
        val_rollout_projected=np.asarray(history["val_rollout_projected"], dtype=np.float32),
        val_lower_sat=np.asarray(history["val_lower_sat"], dtype=np.float32),
        val_clip_frac=np.asarray(history["val_clip_frac"], dtype=np.float32),
        val_stable_score=np.asarray(history["val_stable_score"], dtype=np.float32),
        rollout_truth=viz_rollout_truth.astype(np.float32) if viz_rollout_truth is not None else np.array([]),
        rollout_pred_raw=viz_rollout_pred_raw.astype(np.float32) if viz_rollout_pred_raw is not None else np.array([]),
        rollout_pred_projected=viz_rollout_pred_projected.astype(np.float32) if viz_rollout_pred_projected is not None else np.array([]),
        rollout_ctrl=viz_rollout_ctrl.astype(np.float32) if viz_rollout_ctrl is not None else np.array([]),
        actuator=data_bundle["actuator"].astype(np.float32),
        normalizer_stats=np.array([json.dumps(asdict(normalizer.stats))], dtype=object),
        residual_scale_norm=residual_scale_np.astype(np.float32),
        projection_diagnostics=np.array([json.dumps(proj_summary)], dtype=object),
        test_multi_horizon_diagnostics=np.array([json.dumps(test_multi_horizon_diag)], dtype=object),
    )

    summary = {
        "model_type": model_type,
        "pde_name": pde_name,
        "controlled": True,
        "architecture_version": "risk_aware_long_rollout",
        "best_val_loss": float(best_val),
        "best_val_epoch": int(best_val_epoch),
        "best_rollout_raw": float(best_rollout_raw) if np.isfinite(best_rollout_raw) else None,
        "best_rollout_epoch": int(best_rollout_epoch),
        "best_stable_score": float(best_stable_score) if np.isfinite(best_stable_score) else None,
        "best_stable_epoch": int(best_stable_epoch),
        "selected_checkpoint_tag": selected_tag,
        "selected_checkpoint_epoch": int(selected_epoch),
        **eval_metrics,
        "rollout_rel_final": rollout_rel_final,
        "rollout_rmse_final": rollout_rmse_final,
        "rollout_rel_final_raw": rollout_rel_final_raw,
        "rollout_rmse_final_raw": rollout_rmse_final_raw,
        "rollout_rel_final_projected": rollout_rel_final_projected,
        "rollout_rmse_final_projected": rollout_rmse_final_projected,
        "projection_diagnostics": proj_summary,
        "selected_rollout_diagnostics": selected_rollout_diag,
        "test_multi_horizon_diagnostics": test_multi_horizon_diag,
        "residual_scale_norm": residual_scale_np.astype(float).tolist(),
        "normalizer": asdict(normalizer.stats),
    }
    save_json(summary, os.path.join(save_dir, "summary_metrics.json"))
    print(f"[OK] completed: {model_type}+{pde_name} -> {save_dir}")
    cleanup_memory()
    return summary


# ============================================================
# Data loading/saving
# ============================================================
def save_dataset_npz(path: str, pde_name: str, data: dict):
    if pde_name == "burgers":
        np.savez_compressed(path, state=data["state"], control=data["control"], grid_x=data["grid_x"], actuator=data["actuator"], meta=np.array([json.dumps(data["meta"])], dtype=object))
    else:
        np.savez_compressed(path, state=data["state"], control=data["control"], grid_x=data["grid_x"], grid_y=data["grid_y"], actuator=data["actuator"], meta=np.array([json.dumps(data["meta"])], dtype=object))


def load_dataset_npz(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"][0]))
    out = {"state": z["state"], "control": z["control"], "grid_x": z["grid_x"], "actuator": z["actuator"], "meta": meta}
    if "grid_y" in z.files:
        out["grid_y"] = z["grid_y"]
    return out


def get_or_make_dataset(pde_name: str, result_root: str, force_regenerate: bool = False) -> dict:
    fname = os.path.join(result_root, f"controlled_{pde_name}_dataset.npz")
    if (not force_regenerate) and os.path.exists(fname):
        print(f"[Info] load existing dataset: {fname}")
        return load_dataset_npz(fname)
    if pde_name == "burgers":
        data = generate_burgers_controlled_dataset(CONFIG)
    elif pde_name == "grayscott":
        data = generate_grayscott_controlled_dataset(CONFIG)
    elif pde_name == "kolmogorov":
        data = generate_kolmogorov_controlled_dataset(CONFIG)
    else:
        raise ValueError(pde_name)
    save_dataset_npz(fname, pde_name, data)
    print(f"[OK] saved dataset: {fname}")
    return data


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=str, default=CONFIG["result_root"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--pdes", nargs="+", default=["burgers", "grayscott", "kolmogorov"], choices=["burgers", "grayscott", "kolmogorov"])
    parser.add_argument("--models", nargs="+", default=["fno", "pino", "deeponet", "pinn"], choices=["fno", "pino", "deeponet", "pinn"])
    parser.add_argument("--rollout_k_max", type=int, default=-1,
                        help="Optional global rollout K max for all PDEs. If negative, keep PDE-specific defaults.")
    parser.add_argument("--rollout_k_max_burgers", type=int, default=-1)
    parser.add_argument("--rollout_k_max_grayscott", type=int, default=-1)
    parser.add_argument("--rollout_k_max_kolmogorov", type=int, default=-1)
    parser.add_argument("--rollout_k_start", type=int, default=CONFIG["rollout_k_start"])
    parser.add_argument("--rollout_warmup_epochs", type=int, default=-1,
                        help="Override rollout-loss warmup. If negative, NC12Boost presets choose per PDE/model values.")
    parser.add_argument("--rollout_loss_ramp_epochs", type=int, default=-1,
                        help="Override rollout-loss ramp. If negative, NC12Boost presets choose per PDE/model values.")
    parser.add_argument("--rollout_val_eval_period", type=int, default=CONFIG["rollout_val_eval_period"])
    parser.add_argument("--selection_rollout_warmup_epochs", type=int, default=-1,
                        help="Epoch before validation multi-horizon rollout selection starts. The default is min(CONFIG value, epochs//3) for short diagnostic runs.")
    parser.add_argument("--force_regenerate_data", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--disable_nc12boost", action="store_true",
                        help="Disable NC12Boost architecture/training presets and recover improved.py behavior as much as possible.")
    parser.add_argument("--repair_only", action="store_true", default=True,
                        help="Skip already-good pairs if they are accidentally included in --pdes/--models. Enabled by default in this repair script.")
    parser.add_argument("--no_repair_only", action="store_false", dest="repair_only",
                        help="Allow training non-repair pairs as well.")
    parser.add_argument("--disable_residual_step_limit", action="store_true",
                        help="Disable tanh residual step-size limiter.")
    parser.add_argument("--residual_limit_multiplier", type=float, default=-1.0,
                        help="Override multiplier for data-derived residual scale. If negative, use preset/default.")
    parser.add_argument("--grayscott_bound_w_low", type=float, default=CONFIG["bound_loss_weight_lower"]["grayscott"],
                        help="Soft lower-bound loss weight for Gray--Scott concentrations.")
    parser.add_argument("--grayscott_bound_w_high", type=float, default=CONFIG["bound_loss_weight_upper"]["grayscott"],
                        help="Soft upper-bound loss weight for Gray--Scott concentrations.")
    parser.add_argument("--no_projected_rollout_eval", action="store_true",
                        help="Disable diagnostic projected rollout evaluation.")
    parser.add_argument("--rollout_train_every", type=int, default=CONFIG["rollout_train_every"],
                        help="Compute long rollout loss every N mini-batches; larger is faster.")
    parser.add_argument("--rollout_train_max_batch", type=int, default=CONFIG["rollout_train_max_batch"],
                        help="Subsample at most this many samples per long-rollout mini-batch.")
    parser.add_argument("--risk_rollout_weight", type=float, default=-1.0,
                        help="Override risk-rollout weight for all selected PDE/model pairs. Use negative to keep model-specific defaults.")
    parser.add_argument("--grad_rollout_weight", type=float, default=-1.0,
                        help="Override gradient-rollout weight for all selected PDE/model pairs. Use negative to keep model-specific defaults.")
    parser.add_argument("--disable_universal_local_refine_2d", action="store_true",
                        help="Disable universal local residual refinement for Gray--Scott FNO/PINO/PINN.")
    parser.add_argument("--rollout_select_horizons", type=str, default=",".join(map(str, CONFIG["rollout_select_horizons"])),
                        help="Comma-separated horizons used for validation checkpoint selection.")
    parser.add_argument("--num_val_rollout_eval_traj", type=int, default=CONFIG["num_val_rollout_eval_traj"])
    parser.add_argument("--num_rollout_eval_traj", type=int, default=CONFIG["num_rollout_eval_traj"])
    parser.add_argument("--num_workers", type=int, default=CONFIG["num_workers"])
    parser.add_argument("--disable_grayscott_bounded_state", action="store_true",
                        help="Disable differentiable Gray--Scott bounded-state prediction.")
    parser.add_argument("--soft_clamp_beta", type=float, default=CONFIG["soft_clamp_beta"])
    parser.add_argument("--input_noise_std_norm", type=float, default=CONFIG["input_noise_std_norm"])
    args = parser.parse_args()

    CONFIG["result_root"] = args.result_root
    CONFIG["epochs"] = int(args.epochs)
    CONFIG["lr"] = float(args.lr)
    CONFIG["rollout_k_start"] = int(args.rollout_k_start)
    if int(args.rollout_k_max) >= 0:
        CONFIG["rollout_k_max"] = {"burgers": int(args.rollout_k_max), "grayscott": int(args.rollout_k_max), "kolmogorov": int(args.rollout_k_max)}
    else:
        CONFIG["rollout_k_max"] = dict(CONFIG["rollout_k_max"])
    if int(args.rollout_k_max_burgers) >= 0:
        CONFIG["rollout_k_max"]["burgers"] = int(args.rollout_k_max_burgers)
    if int(args.rollout_k_max_grayscott) >= 0:
        CONFIG["rollout_k_max"]["grayscott"] = int(args.rollout_k_max_grayscott)
    if int(args.rollout_k_max_kolmogorov) >= 0:
        CONFIG["rollout_k_max"]["kolmogorov"] = int(args.rollout_k_max_kolmogorov)
    CONFIG["rollout_val_eval_period"] = int(args.rollout_val_eval_period)
    if int(args.rollout_warmup_epochs) >= 0:
        CONFIG["rollout_warmup_epochs"] = int(args.rollout_warmup_epochs)
    if int(args.rollout_loss_ramp_epochs) >= 0:
        CONFIG["rollout_loss_ramp_epochs"] = int(args.rollout_loss_ramp_epochs)
    if int(args.selection_rollout_warmup_epochs) >= 0:
        CONFIG["selection_rollout_warmup_epochs"] = int(args.selection_rollout_warmup_epochs)
    else:
        # In short diagnostic runs (e.g. 80 epochs), a fixed warmup of 70 gives at
        # most one validation rollout. Start earlier so H40/H60/H80 trends are
        # visible, while keeping the original behavior for long runs.
        CONFIG["selection_rollout_warmup_epochs"] = min(
            int(CONFIG.get("selection_rollout_warmup_epochs", 70)),
            max(10, int(CONFIG["epochs"]) // 3),
        )
    CONFIG["amp"] = not args.no_amp
    CONFIG["nc12boost_enabled"] = not bool(args.disable_nc12boost)
    CONFIG["residual_step_limit"] = not args.disable_residual_step_limit
    if float(args.residual_limit_multiplier) >= 0.0:
        CONFIG["residual_limit_multiplier"] = float(args.residual_limit_multiplier)
    CONFIG["bound_loss_weight_lower"]["grayscott"] = float(args.grayscott_bound_w_low)
    CONFIG["bound_loss_weight_upper"]["grayscott"] = float(args.grayscott_bound_w_high)
    CONFIG["evaluate_projected_rollout"] = not args.no_projected_rollout_eval
    CONFIG["rollout_train_every"] = int(args.rollout_train_every)
    CONFIG["rollout_train_max_batch"] = int(args.rollout_train_max_batch)
    if float(args.risk_rollout_weight) >= 0.0:
        CONFIG["risk_rollout_weight"] = {"burgers": float(args.risk_rollout_weight), "grayscott": float(args.risk_rollout_weight), "kolmogorov": float(args.risk_rollout_weight)}
    if float(args.grad_rollout_weight) >= 0.0:
        CONFIG["grad_rollout_weight"] = {"burgers": float(args.grad_rollout_weight), "grayscott": float(args.grad_rollout_weight), "kolmogorov": float(args.grad_rollout_weight)}
    CONFIG["rollout_select_horizons"] = [int(x) for x in str(args.rollout_select_horizons).split(",") if str(x).strip()]
    CONFIG["num_val_rollout_eval_traj"] = int(args.num_val_rollout_eval_traj)
    CONFIG["num_rollout_eval_traj"] = int(args.num_rollout_eval_traj)
    CONFIG["num_workers"] = int(args.num_workers)
    CONFIG["grayscott_bounded_state"] = not bool(args.disable_grayscott_bounded_state)
    CONFIG["soft_clamp_beta"] = float(args.soft_clamp_beta)
    CONFIG["input_noise_std_norm"] = float(args.input_noise_std_norm)
    CONFIG["universal_local_refine_2d"] = not bool(args.disable_universal_local_refine_2d)
    if os.environ.get("DEEP_ENSEMBLE_SEED"):
        CONFIG["seed"] = int(os.environ["DEEP_ENSEMBLE_SEED"])
    if os.environ.get("BATCH_SIZE_MULT"):
        mult = int(os.environ["BATCH_SIZE_MULT"])
        CONFIG["operator_batch_size_2d"] *= mult
        CONFIG["coord_batch_size_2d"] *= mult
    set_seed(CONFIG["seed"])
    ensure_dir(CONFIG["result_root"])

    # Store the command-line-resolved base configuration. Each PDE/model run
    # starts from this base and then applies NC12Boost only if intended. This
    # prevents settings from Burgers or Kolmogorov leaking into protected
    # Gray--Scott DeepONet/PINO/FNO runs.
    runtime_base_config = copy.deepcopy(CONFIG)
    save_json(CONFIG, os.path.join(CONFIG["result_root"], "training_configuration.json"))

    all_data = {}
    for pde in args.pdes:
        all_data[pde] = get_or_make_dataset(pde, CONFIG["result_root"], args.force_regenerate_data)

    summary_all = {}
    for pde_name in args.pdes:
        data = all_data[pde_name]
        n_traj = data["state"].shape[0]
        train_ids, val_ids, test_ids = split_traj_ids(n_traj, CONFIG["train_ratio"], CONFIG["val_ratio"], CONFIG["seed"])
        for model_type in args.models:
            if bool(args.repair_only) and (not is_repair_pair(pde_name, model_type)):
                print(f"[Skip] {model_type}+{pde_name} is not in REPAIR_PAIRS; existing good result should be kept.")
                continue
            # Reset mutable global settings, then apply per PDE/model training
            # presets. Explicit command-line overrides are re-applied after the
            # preset so user choices remain respected.
            CONFIG.clear()
            CONFIG.update(copy.deepcopy(runtime_base_config))
            apply_model_training_preset(pde_name, model_type)
            if int(args.rollout_warmup_epochs) >= 0:
                CONFIG["rollout_warmup_epochs"] = int(args.rollout_warmup_epochs)
            if int(args.rollout_loss_ramp_epochs) >= 0:
                CONFIG["rollout_loss_ramp_epochs"] = int(args.rollout_loss_ramp_epochs)
            if float(args.residual_limit_multiplier) >= 0.0:
                CONFIG["residual_limit_multiplier"] = float(args.residual_limit_multiplier)
            metrics = train_one_model(model_type, pde_name, data, train_ids, val_ids, test_ids, CONFIG["result_root"])
            summary_all[f"{pde_name}/{model_type}"] = metrics

    save_json(summary_all, os.path.join(CONFIG["result_root"], "training_summary.json"))
    summary_txt = os.path.join(CONFIG["result_root"], "training_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== Controlled PDE surrogate summary metrics ===\n")
        for key, val in summary_all.items():
            f.write(
                f"{key}: rel_l2_mean={val['rel_l2_mean']:.6f}, rel_l2_q95={val['rel_l2_q95']:.6f}, "
                f"mae_mean={val['mae_mean']:.6f}, risk_spearman={val['risk_spearman']:.4f}, "
                f"rollout_rel_final={val['rollout_rel_final']:.6f}, best_val_loss={val['best_val_loss']:.6e}, selected={val['selected_checkpoint_tag']}@{val['selected_checkpoint_epoch']}\n"
            )
    print("\n" + "=" * 86)
    print("全部受控 surrogate V2 训练完成！")
    print(f"结果目录: {os.path.abspath(CONFIG['result_root'])}")
    print(f"汇总文本: {summary_txt}")
    print("=" * 86)


if __name__ == "__main__":
    main()

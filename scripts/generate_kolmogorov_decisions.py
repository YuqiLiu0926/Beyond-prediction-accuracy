# -*- coding: utf-8 -*-
# generate_kolmogorov_decisions.py

from __future__ import annotations

import os
import json
import math
import argparse
import importlib.util
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Any, Dict

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2, irfft2


# ============================================================
# Global
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] device = {DEVICE}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Model definitions (must match train_all_models_controlled.py)
# ============================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul(self, x_ft, weight):
        return torch.einsum("bixy,ioxy->boxy", x_ft, weight)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        m1 = min(self.modes1, x_ft.shape[-2])
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft[:, :, :m1, :m2] = self.compl_mul(
            x_ft[:, :, :m1, :m2], self.weight[:, :, :m1, :m2]
        )
        return irfft2(out_ft, s=(H, W), dim=(-2, -1))


class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width=24, modes=12, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.specs = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, 1),
        )

    def forward(self, x):
        x = self.lift(x)
        for spec, w in zip(self.specs, self.ws):
            x = F.gelu(spec(x) + w(x))
        return self.proj(x)


class PINN2d(nn.Module):
    def __init__(self, branch_dim: int, out_channels: int, hidden: int = 256, depth: int = 4):
        super().__init__()
        dims = [branch_dim + 2] + [hidden] * depth + [out_channels]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, branch_rep: torch.Tensor, coords: torch.Tensor):
        inp = torch.cat([branch_rep, coords], dim=-1)
        out = self.net(inp)
        return out.permute(0, 3, 1, 2).contiguous()


class DeepONet2d(nn.Module):
    def __init__(self, branch_dim: int, out_channels: int, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.out_channels = out_channels

        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden * out_channels),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, branch: torch.Tensor, coords: torch.Tensor):
        B, H, W, _ = coords.shape
        b = self.branch(branch).view(B, self.out_channels, self.hidden)
        t = self.trunk(coords.view(B, H * W, 2))
        out = torch.einsum("bch,bnh->bcn", b, t) + self.bias[None, :, None]
        return out.view(B, self.out_channels, H, W)


# ============================================================
# Config
# ============================================================
@dataclass
class CFG:
    # I/O
    train_script: str = "./scripts/train_discovery_surrogates.py"
    ckpt_path: str = "./external_data/checkpoints/discovery/kolmogorov/deeponet.pt"
    dataset_path: str = "./external_data/datasets/discovery/kolmogorov_controlled_trajectories.npz"
    outdir: str = "./outputs/decision_archives/kolmogorov/deeponet/challenge_conditions"

    seed: int = 42

    # true env refinement
    dt_true: float = 0.0025

    # learned-surrogate control optimization
    H: int = 100
    n_eval_ic: int = 80

    # hard / OOD IC selection
    ic_mode: str = "ood_amplified"   # random, high_gradient, ood_amplified, challenge_conditions
    candidate_pool_mult: int = 4
    ood_amp_scale: float = 1.22
    ood_hf_amp: float = 0.25
    ood_hf_mode_x: int = 3
    ood_hf_mode_y: int = 2

    # control blocking + sequential linearization
    block_len: int = 5
    n_slp_iters: int = 3
    fd_eps_u: float = 3e-2
    trust_radius: float = 0.15
    blend: float = 1.0

    # control bounds
    umin: float = -0.50
    umax: float = 0.50

    # performance-seeking objective + nominal safety constraint
    # maximize actuator-region vorticity response under nominal safety certificate
    w_perf_terminal: float = 8.0
    w_perf_stage: float = 1.5
    w_ctrl: float = 1e-5
    w_du: float = 2e-4

    # surrogate safety budget
    z_limit_scale: float = 1.02
    z_limit_margin: float = 0.00
    boundary_mode: str = "surrogate_zero"

    dpi: int = 300

    ipopt_print_level: int = 0  # 0 disables output; 1-12 increase verbosity.
    max_iter: int = 1000        # Maximum IPOPT iterations.


# ============================================================
# Utilities
# ============================================================
def make_periodic_2d_grid(nx: int, ny: int, Lx: float, Ly: float):
    x = np.linspace(0.0, Lx, nx, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, Ly, ny, endpoint=False, dtype=np.float64)
    dx = Lx / nx
    dy = Ly / ny
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y, dx, dy


def gaussian_profile_2d(X: np.ndarray, Y: np.ndarray, cx: float, cy: float, sigma: float, Lx: float, Ly: float):
    dx = np.minimum(np.abs(X - cx), Lx - np.abs(X - cx))
    dy = np.minimum(np.abs(Y - cy), Ly - np.abs(Y - cy))
    G = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    G = G / (np.max(np.abs(G)) + 1e-12)
    return G.astype(np.float64)


def grad_mag_2d(z: np.ndarray, dx: float, dy: float):
    gx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2.0 * dx)
    gy = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2.0 * dy)
    return np.sqrt(gx**2 + gy**2)


def base_kolmogorov_field(X: np.ndarray, Y: np.ndarray, forcing_k: int) -> np.ndarray:
    return forcing_k * np.cos(forcing_k * Y)


def kolmogorov_step(omega: np.ndarray, a: float, gamma: np.ndarray,
                    nu: float, dt: float, forcing_amp: float, forcing_k: int,
                    x: np.ndarray, y: np.ndarray):
    """
    omega_t + u·grad(omega) = nu Delta omega + forcing + a * gamma
    periodic pseudo-spectral RK4
    """
    ny, nx = omega.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    invK2 = np.zeros_like(K2)
    mask = K2 > 0
    invK2[mask] = 1.0 / K2[mask]

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

        adv = u * wx + v * wy
        return -adv + nu * lap + base_forcing + a * gamma

    k1 = rhs(omega)
    k2 = rhs(omega + 0.5 * dt * k1)
    k3 = rhs(omega + 0.5 * dt * k2)
    k4 = rhs(omega + dt * k3)
    wn = omega + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return wn.astype(np.float64)


def initial_risk_z(state_w: np.ndarray, dx: float, dy: float) -> float:
    w = state_w[0]
    return float(np.max(grad_mag_2d(w, dx, dy)))


def terminal_response(state_w: np.ndarray, actuator: np.ndarray) -> float:
    w = state_w[0]
    return float(np.mean((w ** 2) * actuator))



# ============================================================
# Frozen / frozen-model-compatible surrogate loader
# ============================================================
def load_module_from_path(module_name: str, file_path: str):
    file_path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"train_script not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _normalizer_from_ckpt(train_mod, ckpt: dict):
    if "normalizer" not in ckpt:
        raise KeyError("Frozen checkpoint does not contain 'normalizer'.")
    stats = ckpt["normalizer"]
    if hasattr(train_mod, "Normalizer") and hasattr(train_mod, "NormalizerStats"):
        return train_mod.Normalizer(train_mod.NormalizerStats(**stats))
    raise AttributeError("train_script must expose Normalizer and NormalizerStats.")


class NC12GridSurrogate:
    """Frozen-surrogate wrapper for controlled two-dimensional PDE decisions.

    It uses the same residual-learning, normalization and
    rollout path as the training/evaluation code, instead of the obsolete
    in-file FNO/PINN/DeepONet definitions.
    """
    def __init__(self, train_mod, model: nn.Module, model_type: str, pde_name: str,
                 normalizer, residual_scale_norm, coords_2d_np: np.ndarray):
        self.train_mod = train_mod
        self.model = model
        self.model_type = str(model_type).lower()
        self.pde_name = str(pde_name).lower()
        self.normalizer = normalizer
        self.residual_scale_np = None if residual_scale_norm is None else np.asarray(residual_scale_norm, dtype=np.float32)
        self.coords_2d = torch.from_numpy(np.asarray(coords_2d_np, dtype=np.float32))
        if hasattr(train_mod, "fourier_features_2d"):
            freqs = list(getattr(train_mod, "CONFIG", {}).get("fourier_freqs", [1, 2, 4, 8, 16]))
            self.coord_feat_2d = train_mod.fourier_features_2d(self.coords_2d, freqs).float()
        else:
            self.coord_feat_2d = None

    def one_step(self, state_t: np.ndarray, a_t: float) -> np.ndarray:
        state_t = np.asarray(state_t, dtype=np.float32)
        s_norm_np = self.normalizer.normalize_state_np(state_t)
        curr = torch.from_numpy(s_norm_np[None].astype(np.float32)).to(DEVICE)
        a_phys = torch.tensor([[float(a_t)]], dtype=torch.float32, device=DEVICE)
        a_norm = self.normalizer.norm_control_t(a_phys)
        with torch.no_grad():
            nxt_norm, _ = self.train_mod.predict_next_norm(
                self.model, self.model_type, self.pde_name,
                curr, a_norm, self.normalizer,
                x_norm_1d=None,
                coords_2d=self.coords_2d.to(DEVICE),
                coord_feat_1d=None,
                coord_feat_2d=self.coord_feat_2d.to(DEVICE) if self.coord_feat_2d is not None else None,
                residual_scale_np=self.residual_scale_np,
            )
            nxt_phys = self.normalizer.denorm_state_t(nxt_norm).detach().cpu().numpy()[0]
        return np.asarray(nxt_phys, dtype=np.float64)

    def rollout(self, s0: np.ndarray, a_seq: np.ndarray) -> np.ndarray:
        if hasattr(self.train_mod, "rollout_model_np"):
            arr = self.train_mod.rollout_model_np(
                self.model, self.model_type, self.pde_name,
                np.asarray(s0, dtype=np.float32),
                np.asarray(a_seq, dtype=np.float32),
                self.normalizer,
                None,
                self.coords_2d,
                None,
                self.coord_feat_2d,
                residual_scale_np=self.residual_scale_np,
                projected=False,
                return_diag=False,
            )
            return np.asarray(arr, dtype=np.float64)
        s = np.asarray(s0, dtype=np.float64).copy()
        states = [s.copy()]
        for a in a_seq:
            s = self.one_step(s, float(a))
            states.append(s.copy())
        return np.stack(states, axis=0)


def build_model_from_ckpt(ckpt: dict, train_script: Optional[str] = None, coords_2d: Optional[np.ndarray] = None,
                          expected_pde: str = "kolmogorov"):
    train_script = train_script or CFG.train_script
    train_mod = load_module_from_path(f"{expected_pde}_frozen_model_training_module", train_script)

    cfg = ckpt["cfg"]
    exp = cfg.get("experiment", {})
    data_meta = cfg.get("data_meta", {})
    pde_name = str(ckpt.get("pde_name", exp.get("pde", expected_pde))).lower()
    model_type = str(ckpt.get("model_type", exp.get("mode", ""))).lower()
    if pde_name != expected_pde:
        raise RuntimeError(f"The {expected_pde} decision generator received pde_name={pde_name}.")
    if not bool(exp.get("controlled", True)):
        raise RuntimeError(f"This data-only script supports controlled {expected_pde} checkpoints only.")

    model_cfg = cfg.get("model_config", {})
    state_shape = tuple(model_cfg.get("state_shape", [
        int(data_meta.get("channels", 2 if expected_pde == "grayscott" else 1)),
        int(data_meta.get("ny", data_meta.get("nx", 48))),
        int(data_meta.get("nx", data_meta.get("ny", 48))),
    ]))
    coord_dim = int(model_cfg.get("coord_dim", 2 + 4 * len(model_cfg.get("fourier_freqs", [1, 2, 4, 8, 16]))))

    if hasattr(train_mod, "apply_model_training_preset"):
        try:
            train_mod.apply_model_training_preset(pde_name, model_type)
        except Exception as e:
            print(f"[Warn] apply_model_training_preset failed but continuing: {e}")

    model = train_mod.get_model(model_type, pde_name, state_shape, coord_dim)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(DEVICE)

    if coords_2d is None:
        nx = int(data_meta["nx"])
        ny = int(data_meta["ny"])
        Lx = float(data_meta["Lx"])
        Ly = float(data_meta["Ly"])
        x = np.linspace(0.0, Lx, nx, endpoint=False, dtype=np.float64)
        y = np.linspace(0.0, Ly, ny, endpoint=False, dtype=np.float64)
        x_norm = 2.0 * (x / Lx) - 1.0
        y_norm = 2.0 * (y / Ly) - 1.0
        Xn, Yn = np.meshgrid(x_norm, y_norm)
        coords_2d = np.stack([Xn, Yn], axis=-1).astype(np.float32)

    normalizer = _normalizer_from_ckpt(train_mod, ckpt)
    residual_scale_norm = ckpt.get("residual_scale_norm", None)

    surrogate = NC12GridSurrogate(train_mod, model, model_type, pde_name, normalizer, residual_scale_norm, coords_2d)
    surrogate.raw_model = model
    surrogate.ckpt_meta = {
        "architecture_version": ckpt.get("architecture_version", ""),
        "selected_checkpoint_tag": ckpt.get("selected_checkpoint_tag", ""),
        "selected_checkpoint_epoch": ckpt.get("selected_checkpoint_epoch", ""),
        "has_normalizer": "normalizer" in ckpt,
        "has_residual_scale_norm": "residual_scale_norm" in ckpt,
        "train_script": train_script,
    }
    return surrogate, model_type, data_meta


def controlled_surrogate_one_step(model: nn.Module, model_type: str,
                                  state_t: np.ndarray, a_t: float,
                                  coords_2d: np.ndarray) -> np.ndarray:
    if hasattr(model, "one_step"):
        return model.one_step(state_t, float(a_t))
    raise RuntimeError(
        "The compatibility-only in-file surrogate is disabled for frozen runs. "
        "Use build_model_from_ckpt(..., train_script=...) to construct NC12GridSurrogate."
    )


def surrogate_rollout(model: nn.Module, model_type: str,
                      s0: np.ndarray, a_seq: np.ndarray, coords_2d: np.ndarray) -> np.ndarray:
    if hasattr(model, "rollout"):
        return model.rollout(s0, a_seq)
    s = s0.copy().astype(np.float64)
    states = [s.copy()]
    for a in a_seq:
        s = controlled_surrogate_one_step(model, model_type, s, float(a), coords_2d)
        states.append(s.copy())
    return np.stack(states, axis=0)   # [T+1,C,H,W]


# ============================================================
# Hard / OOD IC selection
# ============================================================
def make_ood_variant(state0: np.ndarray, X: np.ndarray, Y: np.ndarray, base_field: np.ndarray,
                     cfg: CFG, rng: np.random.Generator) -> np.ndarray:
    """
    Amplify deviation from base Kolmogorov flow and add a high-frequency perturbation.
    """
    s = state0.copy()
    w = base_field + cfg.ood_amp_scale * (s[0] - base_field)

    phase_x = rng.uniform(0.0, 2.0 * np.pi)
    phase_y = rng.uniform(0.0, 2.0 * np.pi)
    hf = np.sin(cfg.ood_hf_mode_x * X + phase_x) * np.cos(cfg.ood_hf_mode_y * Y + phase_y)
    hf = hf / (np.max(np.abs(hf)) + 1e-12)
    w = w + cfg.ood_hf_amp * hf

    return w[None].astype(np.float64)


def load_initial_conditions_from_dataset(dataset_path: str,
                                         ckpt_cfg: dict,
                                         n_eval_ic: int,
                                         X: np.ndarray, Y: np.ndarray,
                                         dx: float, dy: float,
                                         base_field: np.ndarray,
                                         cfg: CFG) -> Tuple[np.ndarray, List[str], np.ndarray]:
    data = np.load(dataset_path, allow_pickle=True)
    state = data["state"]   # [J,T+1,1,H,W]
    split_info = ckpt_cfg.get("split_info", {})

    if "test_ids" in split_info and len(split_info["test_ids"]) > 0:
        test_ids = np.array(split_info["test_ids"], dtype=int)
    else:
        J = state.shape[0]
        n_train = int(round(0.7 * J))
        n_val = int(round(0.1 * J))
        test_ids = np.arange(n_train + n_val, J)

    s0_pool = state[test_ids, 0].astype(np.float64)
    z0_pool = np.asarray([initial_risk_z(s, dx, dy) for s in s0_pool], dtype=np.float64)

    order = np.argsort(-z0_pool)
    pool_size = min(len(order), max(n_eval_ic * cfg.candidate_pool_mult, n_eval_ic))
    top_idx = order[:pool_size]
    s0_top = s0_pool[top_idx]
    z0_top = z0_pool[top_idx]

    rng = np.random.default_rng(cfg.seed)

    if cfg.ic_mode == "random":
        ids = rng.choice(len(s0_pool), size=min(n_eval_ic, len(s0_pool)), replace=False)
        sel = s0_pool[ids]
        z0 = np.asarray([initial_risk_z(s, dx, dy) for s in sel], dtype=np.float64)
        return sel, ["test_random"] * len(ids), z0

    if cfg.ic_mode == "high_gradient":
        take = min(n_eval_ic, len(s0_top))
        return s0_top[:take], ["high_grad"] * take, z0_top[:take]

    if cfg.ic_mode == "ood_amplified":
        take = min(n_eval_ic, len(s0_top))
        out = [make_ood_variant(s0_top[i], X, Y, base_field, cfg, rng) for i in range(take)]
        out = np.stack(out, axis=0)
        z0 = np.asarray([initial_risk_z(s, dx, dy) for s in out], dtype=np.float64)
        return out, ["ood_amp"] * take, z0

    if cfg.ic_mode == "challenge_conditions":
        n_half = max(1, n_eval_ic // 2)
        take_in = min(n_half, len(s0_top))
        take_ood = min(n_eval_ic - take_in, len(s0_top) - take_in)

        out_list: List[np.ndarray] = []
        tags: List[str] = []

        for i in range(take_in):
            out_list.append(s0_top[i].copy())
            tags.append("high_grad")

        for j in range(take_ood):
            base = s0_top[take_in + j]
            out_list.append(make_ood_variant(base, X, Y, base_field, cfg, rng))
            tags.append("ood_amp")

        out = np.stack(out_list, axis=0)
        z0 = np.asarray([initial_risk_z(s, dx, dy) for s in out], dtype=np.float64)
        order2 = np.argsort(-z0)
        out = out[order2]
        z0 = z0[order2]
        tags = [tags[i] for i in order2]
        return out[:n_eval_ic], tags[:n_eval_ic], z0[:n_eval_ic]

    raise ValueError(f"Unknown ic_mode={cfg.ic_mode}")


# ============================================================
# Risk/performance functionals
# ============================================================
def risk_Z_inf(states: np.ndarray, dx: float, dy: float) -> float:
    vals = []
    for s in states:
        vals.append(initial_risk_z(s, dx, dy))
    return float(np.max(vals))


def nominal_performance(states: np.ndarray, actuator: np.ndarray) -> float:
    vals = [terminal_response(s, actuator) for s in states[1:]]
    return float(np.sum(vals))


# ============================================================
# True env rollout
# ============================================================
def rollout_true(s0_true: np.ndarray, a_seq: np.ndarray, actuator: np.ndarray,
                 x: np.ndarray, y: np.ndarray,
                 nu: float, dt_nom: float, dt_true: float,
                 forcing_amp: float, forcing_k: int):
    substeps = max(1, int(round(dt_nom / dt_true)))
    w = s0_true[0].copy()
    states = [w[None].copy()]

    for a in a_seq:
        for _ in range(substeps):
            w = kolmogorov_step(w, float(a), actuator, nu, dt_true, forcing_amp, forcing_k, x, y)
        states.append(w[None].copy())

    return np.stack(states, axis=0)   # [T+1,1,H,W]


# ============================================================
# Control-block SLP in control space
# ============================================================
def expand_block_controls(alpha: np.ndarray, H: int, block_len: int) -> np.ndarray:
    full = np.repeat(alpha, block_len)
    return full[:H].astype(np.float64)


def nominal_metrics_from_alpha(alpha: np.ndarray, H: int, block_len: int,
                               model: nn.Module, model_type: str,
                               s0: np.ndarray, coords_2d: np.ndarray,
                               actuator: np.ndarray, dx: float, dy: float):
    U = expand_block_controls(alpha, H, block_len)
    states = surrogate_rollout(model, model_type, s0, U, coords_2d)
    J = nominal_performance(states, actuator)
    Z = risk_Z_inf(states, dx, dy)
    return J, Z, U, states


def linearize_functionals_wrt_alpha(alpha_ref: np.ndarray, fd_eps_u: float,
                                    H: int, block_len: int,
                                    model: nn.Module, model_type: str,
                                    s0: np.ndarray, coords_2d: np.ndarray,
                                    actuator: np.ndarray, dx: float, dy: float):
    J_ref, Z_ref, _, _ = nominal_metrics_from_alpha(
        alpha_ref, H, block_len, model, model_type, s0, coords_2d, actuator, dx, dy
    )

    gJ = np.zeros_like(alpha_ref, dtype=np.float64)
    gZ = np.zeros_like(alpha_ref, dtype=np.float64)

    for i in range(len(alpha_ref)):
        ap = alpha_ref.copy()
        am = alpha_ref.copy()
        ap[i] += fd_eps_u
        am[i] -= fd_eps_u

        Jp, Zp, _, _ = nominal_metrics_from_alpha(
            ap, H, block_len, model, model_type, s0, coords_2d, actuator, dx, dy
        )
        Jm, Zm, _, _ = nominal_metrics_from_alpha(
            am, H, block_len, model, model_type, s0, coords_2d, actuator, dx, dy
        )

        gJ[i] = (Jp - Jm) / (2.0 * fd_eps_u)
        gZ[i] = (Zp - Zm) / (2.0 * fd_eps_u)

    return J_ref, Z_ref, gJ, gZ


def solve_linearized_control_problem(alpha_ref: np.ndarray,
                                     J_ref: float, Z_ref: float,
                                     gJ: np.ndarray, gZ: np.ndarray,
                                     z_limit_nom: float,
                                     cfg: CFG):
    nb = len(alpha_ref)
    A = ca.SX.sym("A", nb)

    dA = A - ca.DM(alpha_ref)
    J_lin = J_ref + ca.dot(ca.DM(gJ), dA)
    Z_lin = Z_ref + ca.dot(ca.DM(gZ), dA)

    obj = -J_lin
    obj += cfg.w_ctrl * ca.sumsqr(A)

    for i in range(nb):
        if i == 0:
            obj += cfg.w_du * A[i] ** 2
        else:
            obj += cfg.w_du * (A[i] - A[i - 1]) ** 2

    g = []
    lbg = []
    ubg = []

    # nominal safety
    g.append(Z_lin)
    lbg.append(-ca.inf)
    ubg.append(z_limit_nom)

    # trust region
    for i in range(nb):
        g.append(A[i] - alpha_ref[i])
        lbg.append(-cfg.trust_radius)
        ubg.append(cfg.trust_radius)

    nlp = {"x": A, "f": obj, "g": ca.vertcat(*g)}
    opts = {
        "ipopt.print_level": cfg.ipopt_print_level,
        "print_time": False,
        "ipopt.max_iter": cfg.max_iter,
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    sol = solver(
        x0=alpha_ref,
        lbx=np.full(nb, cfg.umin),
        ubx=np.full(nb, cfg.umax),
        lbg=np.array(lbg, dtype=np.float64),
        ubg=np.array(ubg, dtype=np.float64),
    )

    alpha_opt = np.array(sol["x"]).reshape(-1)
    f_opt = float(np.array(sol["f"]).reshape(()))
    return alpha_opt, f_opt


def sequential_linearized_programming(s0: np.ndarray,
                                      model: nn.Module, model_type: str,
                                      coords_2d: np.ndarray,
                                      actuator: np.ndarray, dx: float, dy: float,
                                      cfg: CFG,
                                      z_limit_override: float | None = None):
    nb = int(math.ceil(cfg.H / cfg.block_len))
    alpha = np.zeros(nb, dtype=np.float64)

    J0, Z0, _, _ = nominal_metrics_from_alpha(
        alpha, cfg.H, cfg.block_len, model, model_type, s0, coords_2d, actuator, dx, dy
    )
    z_limit_nom = (
        float(z_limit_override)
        if z_limit_override is not None
        else cfg.z_limit_scale * Z0 + cfg.z_limit_margin
    )

    last_info = {
        "J_ref": J0,
        "Z_ref": Z0,
        "z_limit_nom": z_limit_nom,
    }

    for _ in range(cfg.n_slp_iters):
        J_ref, Z_ref, gJ, gZ = linearize_functionals_wrt_alpha(
            alpha, cfg.fd_eps_u,
            cfg.H, cfg.block_len,
            model, model_type,
            s0, coords_2d, actuator, dx, dy,
        )

        alpha_new, _ = solve_linearized_control_problem(
            alpha_ref=alpha,
            J_ref=J_ref,
            Z_ref=Z_ref,
            gJ=gJ,
            gZ=gZ,
            z_limit_nom=z_limit_nom,
            cfg=cfg,
        )

        alpha = (1.0 - cfg.blend) * alpha + cfg.blend * alpha_new
        last_info = {
            "J_ref": J_ref,
            "Z_ref": Z_ref,
            "z_limit_nom": z_limit_nom,
        }

    U_opt = expand_block_controls(alpha, cfg.H, cfg.block_len)
    return alpha, U_opt, last_info



# ============================================================
# Plot-ready data export utilities
# ============================================================
import csv
import shutil
from datetime import datetime


def json_safe(obj):
    """Convert numpy / torch-ish objects into JSON-serializable values."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return str(obj)


def write_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)


def save_per_ic_csv(rows: List[Dict[str, Any]], path: str):
    if len(rows) == 0:
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def risk_inf_series(states: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return np.asarray([initial_risk_z(s, dx, dy) for s in states], dtype=np.float64)


def response_series(states: np.ndarray, actuator: np.ndarray) -> np.ndarray:
    return np.asarray([terminal_response(s, actuator) for s in states[1:]], dtype=np.float64)


def gradient_map_w_channel(state_w: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return grad_mag_2d(state_w[0], dx, dy).astype(np.float32)


def update_top_cases(top_cases: List[Dict[str, Any]], pack: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
    top_cases.append(pack)
    top_cases = sorted(top_cases, key=lambda d: float(d["gap"]), reverse=True)
    return top_cases[:k]


def make_selected_case_maps(states_sur: np.ndarray, states_hf: np.ndarray, dx: float, dy: float, n_times: int = 4):
    ids = np.linspace(0, len(states_sur) - 1, n_times, dtype=int)
    grad_sur = []
    grad_hf = []
    err = []
    for k in ids:
        grad_sur.append(gradient_map_w_channel(states_sur[k], dx, dy))
        grad_hf.append(gradient_map_w_channel(states_hf[k], dx, dy))
        err.append(np.abs(states_hf[k, 0] - states_sur[k, 0]).astype(np.float32))
    return ids, np.stack(grad_sur, axis=0), np.stack(grad_hf, axis=0), np.stack(err, axis=0)


def save_top_cases(top_cases: List[Dict[str, Any]], case_dir: str, prefix: str, cfg: CFG):
    ensure_dir(case_dir)
    case_paths = []
    for rank, c in enumerate(top_cases):
        path = os.path.join(case_dir, f"{prefix}_case_rank{rank:02d}.npz")
        np.savez_compressed(
            path,
            ic_idx=np.array([c["ic_idx"]], dtype=np.int64),
            tag=np.array([c["tag"]], dtype=object),
            pde=np.array([c["pde"]], dtype=object),
            model_type=np.array([c["model_type"]], dtype=object),
            ic_mode=np.array([c["ic_mode"]], dtype=object),
            timestamp=np.array([c["timestamp"]], dtype=object),
            z0=np.array([c["z0"]], dtype=np.float64),
            z_sur=np.array([c["z_sur"]], dtype=np.float64),
            z_hf=np.array([c["z_hf"]], dtype=np.float64),
            gap=np.array([c["gap"]], dtype=np.float64),
            z_limit=np.array([c["z_limit"]], dtype=np.float64),
            J_sur=np.array([c["J_sur"]], dtype=np.float64),
            J_hf=np.array([c["J_hf"]], dtype=np.float64),
            U_opt=np.asarray(c["U_opt"], dtype=np.float32),
            alpha_opt=np.asarray(c["alpha_opt"], dtype=np.float32),
            states_sur=np.asarray(c["states_sur"], dtype=np.float32),
            states_hf=np.asarray(c["states_hf"], dtype=np.float32),
            z_sur_ts=np.asarray(c["z_sur_ts"], dtype=np.float64),
            z_hf_ts=np.asarray(c["z_hf_ts"], dtype=np.float64),
            J_sur_ts=np.asarray(c["J_sur_ts"], dtype=np.float64),
            J_hf_ts=np.asarray(c["J_hf_ts"], dtype=np.float64),
            time=np.arange(len(c["z_sur_ts"]), dtype=np.float64) * float(cfg.dt_nom),
            time_state=np.arange(len(c["z_sur_ts"]), dtype=np.float64) * float(cfg.dt_nom),
            time_control=np.arange(len(c["U_opt"]), dtype=np.float64) * float(cfg.dt_nom),
            selected_time_ids=np.asarray(c["selected_time_ids"], dtype=np.int64),
            grad_sur_selected=np.asarray(c["grad_sur_selected"], dtype=np.float32),
            grad_hf_selected=np.asarray(c["grad_hf_selected"], dtype=np.float32),
            err_selected=np.asarray(c["err_selected"], dtype=np.float32),
        )
        case_paths.append(path)
    return case_paths


def build_run_paths(outroot: str, pde: str, model_type: str, ic_mode: str, timestamp: str, flat_outdir: bool = False):
    run_id = f"surrogate_guided_decision_{pde}_{model_type}_{ic_mode}_{timestamp}"
    run_dir = outroot if flat_outdir else os.path.join(outroot, run_id)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "hardest_cases"))
    prefix = run_id
    return run_id, run_dir, prefix


# ============================================================
# Plots (same style as Burgers enhanced)
# ============================================================
def plot_scatter_nominal_vs_true(z_pred: np.ndarray, z_true: np.ndarray, z_limit: np.ndarray,
                                 out_png: str, cfg: CFG):
    plt.figure(figsize=(5.8, 5.2), dpi=cfg.dpi)
    sc = plt.scatter(z_pred, z_true, c=z_limit, s=34, alpha=0.85, cmap="viridis")

    lo = min(np.min(z_pred), np.min(z_true), np.min(z_limit))
    hi = max(np.max(z_pred), np.max(z_true), np.max(z_limit))
    pad = 0.04 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad

    plt.plot([lo, hi], [lo, hi], "--", label="ideal")
    plt.xlabel(r"learned nominal risk $\hat Z(x_0,u^\star)$")
    plt.ylabel(r"true risk $Z^{true}(x_0,u^\star)$")
    plt.title("Surrogate-guided optimization diagnostics")
    plt.legend(frameon=True)

    cbar = plt.colorbar(sc)
    cbar.set_label(r"nominal safety limit $z_{\max}^{nom}$")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_gap_hist(gaps: np.ndarray, out_png: str, cfg: CFG):
    plt.figure(figsize=(6.2, 4.6), dpi=cfg.dpi)
    plt.hist(gaps, bins=18, alpha=0.85)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.2)
    plt.xlabel(r"$Z^{true} - \hat Z$")
    plt.ylabel("count")
    plt.title("Risk underestimation gap for surrogate-optimized controls")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_violation_hist(excess_true: np.ndarray, excess_nom: np.ndarray, out_png: str, cfg: CFG):
    plt.figure(figsize=(6.6, 4.8), dpi=cfg.dpi)
    plt.hist(excess_nom, bins=18, alpha=0.7, label="nominal excess")
    plt.hist(excess_true, bins=18, alpha=0.7, label="true excess")
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel(r"violation excess relative to $z_{\max}^{nom}$")
    plt.ylabel("count")
    plt.title("Nominal-safe vs true-safe mismatch")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_hardest_case(states_nom: np.ndarray,
                      states_true: np.ndarray,
                      cfg: CFG,
                      out_png: str):
    """
    Compare HF vorticity fields with surrogate contours at four time points.
    """
    ids = np.linspace(0, len(states_nom) - 1, 4, dtype=int)

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.8), dpi=cfg.dpi)
    axs = axs.ravel()

    for ax, k in zip(axs, ids):
        w_true = states_true[k, 0]
        w_nom = states_nom[k, 0]

        im = ax.imshow(w_true, origin="lower", cmap="RdBu_r", alpha=0.90)

        try:
            vmax = max(np.max(np.abs(w_true)), np.max(np.abs(w_nom)))
            lv = np.array([-0.6, -0.3, 0.3, 0.6]) * vmax
            lv = lv[np.abs(lv) > 1e-10]
            ax.contour(w_true, levels=lv, colors="white", linewidths=1.8)
            ax.contour(w_nom, levels=lv, colors="red", linewidths=1.1, linestyles="--")
        except Exception:
            pass

        ax.set_title(f"t = {k * cfg.dt_nom:.4f}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="white", lw=2.0, label="true contour"),
        Line2D([0], [0], color="red", lw=1.5, linestyle="--", label="surrogate contour"),
    ]
    axs[0].legend(handles=handles, frameon=True, loc="upper right")

    fig.suptitle("Hardest optimizer-exploitation case: learned surrogate vs true rollout", y=0.99)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_control_sequence(U_opt: np.ndarray, out_png: str, cfg: CFG):
    plt.figure(figsize=(7.0, 3.8), dpi=cfg.dpi)
    plt.step(np.arange(len(U_opt)), U_opt, where="post")
    plt.xlabel("MPC step")
    plt.ylabel("control amplitude $a_k$")
    plt.title("Optimized open-loop control sequence")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_selected_ic_risks(z0s: np.ndarray, tags: List[str], out_png: str, cfg: CFG):
    colors = []
    for t in tags:
        if t == "high_grad":
            colors.append("tab:red")
        elif t == "ood_amp":
            colors.append("tab:orange")
        else:
            colors.append("tab:blue")

    plt.figure(figsize=(7.2, 4.0), dpi=cfg.dpi)
    plt.bar(np.arange(len(z0s)), z0s, color=colors)
    plt.xlabel("selected IC index")
    plt.ylabel(r"initial risk $Z_0$")
    plt.title(f"Selected IC pool ({cfg.ic_mode})")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_script", type=str, default=CFG.train_script,
                        help="Path to train_discovery_surrogates.py")
    parser.add_argument("--ckpt_path", type=str, default=CFG.ckpt_path)
    parser.add_argument("--dataset_path", type=str, default=CFG.dataset_path)
    parser.add_argument("--outdir", type=str, default=CFG.outdir)
    parser.add_argument("--H", type=int, default=CFG.H)
    parser.add_argument("--n_eval_ic", type=int, default=CFG.n_eval_ic)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--ic_mode", type=str, default=CFG.ic_mode,
                        choices=["random", "high_gradient", "ood_amplified", "challenge_conditions"])
    parser.add_argument("--z_limit_scale", type=float, default=CFG.z_limit_scale)
    parser.add_argument("--boundary_mode", choices=("surrogate_zero", "initial_risk"), default=CFG.boundary_mode)
    parser.add_argument("--block_len", type=int, default=CFG.block_len)
    parser.add_argument("--top_k_cases", type=int, default=5)
    parser.add_argument("--timestamp", type=str, default="", help="Optional manual timestamp/run suffix.")
    parser.add_argument("--flat_outdir", action="store_true", help="Save directly into --outdir instead of a timestamped subdirectory.")
    args = parser.parse_args()

    timestamp = args.timestamp.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = CFG(
        train_script=args.train_script,
        ckpt_path=args.ckpt_path,
        dataset_path=args.dataset_path,
        outdir=args.outdir,
        H=args.H,
        n_eval_ic=args.n_eval_ic,
        seed=args.seed,
        ic_mode=args.ic_mode,
        z_limit_scale=args.z_limit_scale,
        boundary_mode=args.boundary_mode,
        block_len=args.block_len,
    )
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    data_meta = ckpt["cfg"].get("data_meta", {})
    pde = str(ckpt.get("pde_name", ckpt["cfg"].get("experiment", {}).get("pde", "kolmogorov"))).lower()
    if pde != "kolmogorov" or not ckpt["cfg"].get("experiment", {}).get("controlled", True):
        raise RuntimeError("This data-only script currently supports controlled Kolmogorov checkpoints only.")

    nx = int(data_meta["nx"])
    ny = int(data_meta["ny"])
    Lx = float(data_meta["Lx"])
    Ly = float(data_meta["Ly"])
    nu = float(data_meta["nu"])
    dt_nom = float(data_meta["dt"])
    forcing_amp = float(data_meta["forcing_amp"])
    forcing_k = int(data_meta["forcing_k"])

    cfg.dt_nom = dt_nom
    cfg.umin = float(data_meta["u_min"])
    cfg.umax = float(data_meta["u_max"])

    x, y, X, Y, dx, dy = make_periodic_2d_grid(nx, ny, Lx, Ly)
    actuator = gaussian_profile_2d(
        X, Y,
        float(data_meta["ctrl_center_x"]),
        float(data_meta["ctrl_center_y"]),
        float(data_meta["ctrl_sigma"]),
        Lx, Ly
    )

    x_norm = 2.0 * (x / Lx) - 1.0
    y_norm = 2.0 * (y / Ly) - 1.0
    Xn, Yn = np.meshgrid(x_norm, y_norm)
    coords_2d = np.stack([Xn, Yn], axis=-1).astype(np.float32)

    model, model_type, data_meta = build_model_from_ckpt(
        ckpt, train_script=cfg.train_script, coords_2d=coords_2d, expected_pde="kolmogorov"
    )

    run_id, run_dir, prefix = build_run_paths(args.outdir, pde, model_type, cfg.ic_mode, timestamp, bool(args.flat_outdir))
    cfg.outdir = run_dir
    ensure_dir(cfg.outdir)

    base_field = base_kolmogorov_field(X, Y, forcing_k)

    u0s_nom, tags, z0s = load_initial_conditions_from_dataset(
        cfg.dataset_path,
        ckpt["cfg"],
        cfg.n_eval_ic,
        X, Y,
        dx, dy,
        base_field,
        cfg
    )

    z_pred_list, z_true_list, gap_list = [], [], []
    J_pred_list, J_true_list, z_limit_list = [], [], []
    margin_sur_list, margin_hf_list = [], []
    nominal_excess_list, hf_excess_list = [], []
    per_ic_rows: List[Dict[str, Any]] = []
    U_all, alpha_all = [], []
    z_sur_ts_all, z_hf_ts_all = [], []
    J_sur_ts_all, J_hf_ts_all = [], []
    top_cases: List[Dict[str, Any]] = []

    t_start = datetime.now()

    for i, s0_nom in enumerate(u0s_nom):
        print(f"[Run:{run_id}] IC {i+1}/{len(u0s_nom)} | tag={tags[i]} | Z0={z0s[i]:.6f}")

        z_limit_override = (
            cfg.z_limit_scale * float(z0s[i]) + cfg.z_limit_margin
            if cfg.boundary_mode == "initial_risk"
            else None
        )

        alpha_opt, U_opt, info = sequential_linearized_programming(
            s0=s0_nom,
            model=model,
            model_type=model_type,
            coords_2d=coords_2d,
            actuator=actuator,
            dx=dx,
            dy=dy,
            cfg=cfg,
            z_limit_override=z_limit_override,
        )

        states_nom = surrogate_rollout(
            model=model,
            model_type=model_type,
            s0=s0_nom,
            a_seq=U_opt,
            coords_2d=coords_2d,
        )
        z_pred = risk_Z_inf(states_nom, dx, dy)
        J_pred = nominal_performance(states_nom, actuator)

        states_true = rollout_true(
            s0_true=s0_nom,
            a_seq=U_opt,
            actuator=actuator,
            x=x,
            y=y,
            nu=nu,
            dt_nom=dt_nom,
            dt_true=cfg.dt_true,
            forcing_amp=forcing_amp,
            forcing_k=forcing_k,
        )
        z_true = risk_Z_inf(states_true, dx, dy)
        J_true = nominal_performance(states_true, actuator)

        z_sur_ts = risk_inf_series(states_nom, dx, dy)
        z_hf_ts = risk_inf_series(states_true, dx, dy)
        J_sur_ts = response_series(states_nom, actuator)
        J_hf_ts = response_series(states_true, actuator)

        gap = float(z_true - z_pred)
        z_limit_nom = float(info["z_limit_nom"])
        margin_sur = float(z_limit_nom - z_pred)
        margin_hf = float(z_limit_nom - z_true)
        nominal_excess = float(z_pred - z_limit_nom)
        hf_excess = float(z_true - z_limit_nom)

        z_pred_list.append(float(z_pred))
        z_true_list.append(float(z_true))
        gap_list.append(gap)
        J_pred_list.append(float(J_pred))
        J_true_list.append(float(J_true))
        z_limit_list.append(z_limit_nom)
        margin_sur_list.append(margin_sur)
        margin_hf_list.append(margin_hf)
        nominal_excess_list.append(nominal_excess)
        hf_excess_list.append(hf_excess)
        U_all.append(U_opt.astype(np.float32))
        alpha_all.append(alpha_opt.astype(np.float32))
        z_sur_ts_all.append(z_sur_ts.astype(np.float32))
        z_hf_ts_all.append(z_hf_ts.astype(np.float32))
        J_sur_ts_all.append(J_sur_ts.astype(np.float32))
        J_hf_ts_all.append(J_hf_ts.astype(np.float32))

        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "pde": pde,
            "model_type": model_type,
            "ic_mode": cfg.ic_mode,
            "ic_idx": i,
            "tag": tags[i],
            "z0": float(z0s[i]),
            "z_sur": float(z_pred),
            "z_hf": float(z_true),
            "gap": gap,
            "z_limit": z_limit_nom,
            "margin_sur": margin_sur,
            "margin_hf": margin_hf,
            "nominal_excess": nominal_excess,
            "hf_excess": hf_excess,
            "surrogate_excess": nominal_excess,
            "true_excess": hf_excess,
            "J_sur": float(J_pred),
            "J_hf": float(J_true),
            "J_gap": float(J_true - J_pred),
            "underestimated": int(gap > 0.0),
            "surrogate_safe": int(z_pred <= z_limit_nom),
            "hf_safe": int(z_true <= z_limit_nom),
            "surrogate_safe_hf_unsafe": int((z_pred <= z_limit_nom) and (z_true > z_limit_nom)),
            "U_mean": float(np.mean(U_opt)),
            "U_std": float(np.std(U_opt)),
            "U_min": float(np.min(U_opt)),
            "U_max": float(np.max(U_opt)),
            "U_tv": float(np.sum(np.abs(np.diff(U_opt)))) if len(U_opt) > 1 else 0.0,
            "alpha_mean": float(np.mean(alpha_opt)),
            "alpha_std": float(np.std(alpha_opt)),
        }
        per_ic_rows.append(row)

        selected_ids, grad_sur_sel, grad_hf_sel, err_sel = make_selected_case_maps(states_nom, states_true, dx, dy, n_times=4)
        top_cases = update_top_cases(top_cases, {
            "ic_idx": i,
            "tag": tags[i],
            "pde": pde,
            "model_type": model_type,
            "ic_mode": cfg.ic_mode,
            "timestamp": timestamp,
            "z0": float(z0s[i]),
            "states_sur": states_nom.copy(),
            "states_hf": states_true.copy(),
            "U_opt": U_opt.copy(),
            "alpha_opt": alpha_opt.copy(),
            "z_sur": float(z_pred),
            "z_hf": float(z_true),
            "gap": gap,
            "z_limit": z_limit_nom,
            "J_sur": float(J_pred),
            "J_hf": float(J_true),
            "z_sur_ts": z_sur_ts.copy(),
            "z_hf_ts": z_hf_ts.copy(),
            "J_sur_ts": J_sur_ts.copy(),
            "J_hf_ts": J_hf_ts.copy(),
            "selected_time_ids": selected_ids,
            "grad_sur_selected": grad_sur_sel,
            "grad_hf_selected": grad_hf_sel,
            "err_selected": err_sel,
        }, k=int(args.top_k_cases))

    z_pred_arr = np.asarray(z_pred_list, dtype=np.float64)
    z_true_arr = np.asarray(z_true_list, dtype=np.float64)
    gap_arr = np.asarray(gap_list, dtype=np.float64)
    J_pred_arr = np.asarray(J_pred_list, dtype=np.float64)
    J_true_arr = np.asarray(J_true_list, dtype=np.float64)
    z_limit_arr = np.asarray(z_limit_list, dtype=np.float64)
    margin_sur_arr = np.asarray(margin_sur_list, dtype=np.float64)
    margin_hf_arr = np.asarray(margin_hf_list, dtype=np.float64)
    nominal_excess_arr = np.asarray(nominal_excess_list, dtype=np.float64)
    hf_excess_arr = np.asarray(hf_excess_list, dtype=np.float64)

    frac = lambda cond: float(np.mean(cond)) if len(gap_arr) else float("nan")
    q = lambda arr, qq: float(np.quantile(arr, qq)) if len(arr) else float("nan")
    pos_gap = gap_arr[gap_arr > 0]
    cvar95_gap = float(np.mean(pos_gap[pos_gap >= np.quantile(pos_gap, 0.95)])) if len(pos_gap) >= 2 else (float(pos_gap[0]) if len(pos_gap) == 1 else 0.0)

    elapsed_sec = (datetime.now() - t_start).total_seconds()

    summary = {
        "experiment_name": "surrogate_guided_decision_evaluation",
        "run_id": run_id,
        "timestamp": timestamp,
        "pde": pde,
        "model_type": model_type,
        "ic_mode": cfg.ic_mode,
        "ckpt_path": cfg.ckpt_path,
        "dataset_path": cfg.dataset_path,
        "outdir": cfg.outdir,
        "n_eval_ic": int(len(u0s_nom)),
        "H": int(cfg.H),
        "block_len": int(cfg.block_len),
        "dt_nom": float(cfg.dt_nom),
        "dt_true": float(cfg.dt_true),
        "elapsed_sec": float(elapsed_sec),
        "mean_initial_risk": float(np.mean(z0s)),
        "mean_surrogate_risk": float(np.mean(z_pred_arr)),
        "mean_hf_risk": float(np.mean(z_true_arr)),
        "mean_gap_hf_minus_surrogate": float(np.mean(gap_arr)),
        "median_gap_hf_minus_surrogate": float(np.median(gap_arr)),
        "q90_gap": q(gap_arr, 0.90),
        "q95_gap": q(gap_arr, 0.95),
        "cvar95_positive_gap": cvar95_gap,
        "fraction_underestimated": frac(gap_arr > 0.0),
        "pearson_r_surrogate_hf_risk": float(np.corrcoef(z_pred_arr, z_true_arr)[0, 1]) if len(z_pred_arr) >= 2 else float("nan"),
        "surrogate_safe_rate": frac(z_pred_arr <= z_limit_arr),
        "hf_safe_rate": frac(z_true_arr <= z_limit_arr),
        "surrogate_safe_hf_unsafe_rate": frac((z_pred_arr <= z_limit_arr) & (z_true_arr > z_limit_arr)),
        "mean_surrogate_violation_excess": float(np.mean(nominal_excess_arr)),
        "mean_hf_violation_excess": float(np.mean(hf_excess_arr)),
        "mean_surrogate_perf": float(np.mean(J_pred_arr)),
        "mean_hf_perf": float(np.mean(J_true_arr)),
        "mean_perf_gap_hf_minus_surrogate": float(np.mean(J_true_arr - J_pred_arr)),
        "worst_gap": float(np.max(gap_arr)),
        "best_gap": float(np.min(gap_arr)),
        "cfg": json_safe(asdict(cfg)),
        "data_meta": json_safe(data_meta),
        "checkpoint_metadata": json_safe(getattr(model, "ckpt_meta", {})),
    }

    prefix_path = os.path.join(cfg.outdir, prefix)
    summary_path = prefix_path + "_summary.json"
    per_ic_path = prefix_path + "_per_ic.csv"
    plot_data_path = prefix_path + "_plot_data.npz"
    manifest_path = os.path.join(cfg.outdir, "run_manifest.json")

    write_json(summary_path, summary)
    save_per_ic_csv(per_ic_rows, per_ic_path)

    top_case_paths = save_top_cases(top_cases, os.path.join(cfg.outdir, "hardest_cases"), prefix, cfg)

    np.savez_compressed(
        plot_data_path,
        run_id=np.array([run_id], dtype=object),
        timestamp=np.array([timestamp], dtype=object),
        pde=np.array([pde], dtype=object),
        model_type=np.array([model_type], dtype=object),
        ic_mode=np.array([cfg.ic_mode], dtype=object),
        tags=np.asarray(tags, dtype=object),
        initial_states=np.asarray(u0s_nom, dtype=np.float32),
        z0=np.asarray(z0s, dtype=np.float32),
        z_sur=z_pred_arr.astype(np.float32),
        z_hf=z_true_arr.astype(np.float32),
        gap=gap_arr.astype(np.float32),
        J_sur=J_pred_arr.astype(np.float32),
        J_hf=J_true_arr.astype(np.float32),
        z_limit=z_limit_arr.astype(np.float32),
        margin_sur=margin_sur_arr.astype(np.float32),
        margin_hf=margin_hf_arr.astype(np.float32),
        surrogate_excess=nominal_excess_arr.astype(np.float32),
        hf_excess=hf_excess_arr.astype(np.float32),
        U_all=np.asarray(U_all, dtype=np.float32),
        alpha_all=np.asarray(alpha_all, dtype=np.float32),
        z_sur_ts_all=np.asarray(z_sur_ts_all, dtype=np.float32),
        z_hf_ts_all=np.asarray(z_hf_ts_all, dtype=np.float32),
        J_sur_ts_all=np.asarray(J_sur_ts_all, dtype=np.float32),
        J_hf_ts_all=np.asarray(J_hf_ts_all, dtype=np.float32),
        time_state=np.arange(cfg.H + 1, dtype=np.float32) * float(cfg.dt_nom),
        time_control=np.arange(cfg.H, dtype=np.float32) * float(cfg.dt_nom),
        config=np.array([json.dumps(json_safe(asdict(cfg)))], dtype=object),
        summary=np.array([json.dumps(json_safe(summary))], dtype=object),
    )

    shutil.copyfile(summary_path, os.path.join(cfg.outdir, "summary.json"))
    shutil.copyfile(per_ic_path, os.path.join(cfg.outdir, "decision_summary.csv"))
    shutil.copyfile(plot_data_path, os.path.join(cfg.outdir, "decision_data.npz"))

    manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "pde": pde,
        "model_type": model_type,
        "ic_mode": cfg.ic_mode,
        "run_dir": cfg.outdir,
        "summary_json": summary_path,
        "per_ic_csv": per_ic_path,
        "plot_data_npz": plot_data_path,
        "stable_summary_json": os.path.join(cfg.outdir, "summary.json"),
        "stable_per_ic_csv": os.path.join(cfg.outdir, "decision_summary.csv"),
        "stable_plot_data_npz": os.path.join(cfg.outdir, "decision_data.npz"),
        "hardest_case_npz": top_case_paths,
    }
    write_json(manifest_path, manifest)

    print("[OK] Kolmogorov decision archive saved to:", cfg.outdir)
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

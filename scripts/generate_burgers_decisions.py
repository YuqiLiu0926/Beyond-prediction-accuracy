# -*- coding: utf-8 -*-
# generate_burgers_decisions.py

from __future__ import annotations

import os
import json
import argparse
import importlib.util
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Any

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft


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
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul(self, x_ft, weight):
        return torch.einsum("bim,iom->bom", x_ft, weight)

    def forward(self, x):
        B, C, N = x.shape
        x_ft = rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        m = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :m] = self.compl_mul(x_ft[:, :, :m], self.weight[:, :, :m])
        return irfft(out_ft, n=N, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, width=32, modes=16, n_layers=4):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.specs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, out_channels, 1),
        )

    def forward(self, x):
        x = self.lift(x)
        for spec, w in zip(self.specs, self.ws):
            x = F.gelu(spec(x) + w(x))
        return self.proj(x)


class PINN1d(nn.Module):
    def __init__(self, branch_dim: int, hidden: int = 128, depth: int = 4):
        super().__init__()
        dims = [branch_dim + 1] + [hidden] * depth + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, branch_rep: torch.Tensor, xcoord: torch.Tensor):
        inp = torch.cat([branch_rep, xcoord], dim=-1)
        out = self.net(inp).squeeze(-1)
        return out


class DeepONet1d(nn.Module):
    def __init__(self, branch_dim: int, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch: torch.Tensor, xcoord: torch.Tensor):
        b = self.branch(branch)
        t = self.trunk(xcoord)
        out = (b[:, None, :] * t).sum(dim=-1) + self.bias
        return out


# ============================================================
# Config
# ============================================================
@dataclass
class CFG:
    # I/O
    train_script: str = "./scripts/train_discovery_surrogates.py"
    ckpt_path: str = "./external_data/checkpoints/discovery/burgers/deeponet.pt"
    dataset_path: str = "./external_data/datasets/discovery/burgers_controlled_trajectories.npz"
    outdir: str = "./outputs/decision_archives/burgers/deeponet/challenge_conditions"

    seed: int = 42

    # true env
    nx_true: int = 256
    dt_true: float = 1.25e-4

    # learned-surrogate MPC
    H: int = 100
    n_eval_ic: int = 80
    u_prev: float = 0.0

    # local linearization around checkpoint surrogate
    eps_x: float = 2e-3
    eps_u: float = 5e-3

    # hard / OOD IC selection
    ic_mode: str = "ood_amplified"   # random, high_gradient, ood_amplified, challenge_conditions
    candidate_pool_mult: int = 4
    ood_amp_scale: float = 1.18
    ood_hf_amp: float = 0.08
    ood_hf_mode: int = 6

    # control bounds (same as training bounds, but optimizer now really uses them)
    umin: float = -0.75
    umax: float = 0.75

    # performance-seeking objective + surrogate safety constraint
    # performance = maximize actuator-region response under nominal safety certificate
    w_perf_terminal: float = 8.0
    w_perf_stage: float = 1.5
    w_energy_reg: float = 1e-4
    w_ctrl: float = 1e-5
    w_du: float = 2e-4

    # surrogate safety budget: Z_hat <= z_limit_nom
    z_limit_scale: float = 1.02
    z_limit_margin: float = 0.00
    boundary_mode: str = "initial_risk"

    # p-norm approximation of ||u_x||_inf
    grad_p: int = 14
    grad_eps: float = 1e-6

    # IPOPT
    ipopt_print_level: int = 0
    max_iter: int = 600

    dpi: int = 300


# ============================================================
# Utilities
# ============================================================
def make_grid(nx: int, L: float):
    return np.linspace(0.0, L, nx, endpoint=False, dtype=np.float64)


def periodic_interp_1d(u_old: np.ndarray, x_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    dx_old = x_old[1] - x_old[0]
    L = x_old[-1] - x_old[0] + dx_old
    x_old_ext = np.concatenate([x_old, [x_old[0] + L]])
    u_old_ext = np.concatenate([u_old, [u_old[0]]])
    xq = ((x_new - x_old[0]) % L) + x_old[0]
    return np.interp(xq, x_old_ext, u_old_ext)


def build_periodic_D1_D2(nx: int, L: float):
    dx = L / nx
    D1 = np.zeros((nx, nx), dtype=np.float64)
    D2 = np.zeros((nx, nx), dtype=np.float64)
    for i in range(nx):
        D1[i, (i + 1) % nx] = 1.0 / (2.0 * dx)
        D1[i, (i - 1) % nx] = -1.0 / (2.0 * dx)

        D2[i, i] = -2.0 / dx**2
        D2[i, (i + 1) % nx] = 1.0 / dx**2
        D2[i, (i - 1) % nx] = 1.0 / dx**2
    return D1, D2, dx


def control_profile(x: np.ndarray, center: float, sigma: float, L: float):
    d = np.minimum(np.abs(x - center), L - np.abs(x - center))
    g = np.exp(-(d**2) / (2.0 * sigma**2))
    g = g / (np.max(np.abs(g)) + 1e-12)
    return g.astype(np.float64)


def burgers_rhs_true(u: np.ndarray, a: float, kappa: np.ndarray, x: np.ndarray, nu: float):
    dx = x[1] - x[0]
    ux = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
    uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx**2)
    return -(u * ux) + nu * uxx + a * kappa


def step_true_ssprk3(u: np.ndarray, a: float, kappa: np.ndarray, x: np.ndarray, nu: float, dt: float):
    f1 = burgers_rhs_true(u, a, kappa, x, nu)
    u1 = u + dt * f1

    f2 = burgers_rhs_true(u1, a, kappa, x, nu)
    u2 = 0.75 * u + 0.25 * (u1 + dt * f2)

    f3 = burgers_rhs_true(u2, a, kappa, x, nu)
    un = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * f3)
    return un.astype(np.float64)


def rollout_true(u0_true: np.ndarray, a_seq: np.ndarray, x_true: np.ndarray, k_true: np.ndarray,
                 nu: float, dt_nom: float, dt_true: float):
    substeps = int(round(dt_nom / dt_true))
    u = u0_true.copy()
    states = [u.copy()]
    for a in a_seq:
        for _ in range(substeps):
            u = step_true_ssprk3(u, float(a), k_true, x_true, nu, dt_true)
        states.append(u.copy())
    return np.stack(states, axis=0)


def risk_Z_inf(states: np.ndarray, dx: float):
    vals = []
    for u in states:
        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
        vals.append(np.max(np.abs(ux)))
    return float(np.max(vals))


def initial_risk_z(u0: np.ndarray, dx: float):
    ux = (np.roll(u0, -1) - np.roll(u0, 1)) / (2.0 * dx)
    return float(np.max(np.abs(ux)))


def terminal_response(u: np.ndarray, kappa: np.ndarray) -> float:
    return float(np.abs(np.dot(kappa, u)) / len(u))



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


class NC12BurgersSurrogate:
    """Load the frozen surrogate through the same inference path used during training."""
    def __init__(self, train_mod, model: nn.Module, model_type: str, normalizer, residual_scale_norm,
                 x_norm_np: np.ndarray):
        self.train_mod = train_mod
        self.model = model
        self.model_type = str(model_type)
        self.pde_name = "burgers"
        self.normalizer = normalizer
        self.residual_scale_np = None if residual_scale_norm is None else np.asarray(residual_scale_norm, dtype=np.float32)
        self.x_norm_1d = torch.from_numpy(np.asarray(x_norm_np, dtype=np.float32))
        if hasattr(train_mod, "fourier_features_1d"):
            freqs = list(getattr(train_mod, "CONFIG", {}).get("fourier_freqs", [1, 2, 4, 8, 16]))
            self.coord_feat_1d = train_mod.fourier_features_1d(self.x_norm_1d, freqs).float()
        else:
            self.coord_feat_1d = None

    def one_step(self, u_t: np.ndarray, a_t: float) -> np.ndarray:
        u_t = np.asarray(u_t, dtype=np.float32).reshape(-1)
        s_norm_np = self.normalizer.normalize_state_np(u_t)  # [1,N]
        curr = torch.from_numpy(s_norm_np[None].astype(np.float32)).to(DEVICE)  # [1,1,N]
        a_phys = torch.tensor([[float(a_t)]], dtype=torch.float32, device=DEVICE)
        a_norm = self.normalizer.norm_control_t(a_phys)
        with torch.no_grad():
            nxt_norm, _ = self.train_mod.predict_next_norm(
                self.model, self.model_type, self.pde_name,
                curr, a_norm, self.normalizer,
                x_norm_1d=self.x_norm_1d.to(DEVICE),
                coords_2d=None,
                coord_feat_1d=self.coord_feat_1d.to(DEVICE) if self.coord_feat_1d is not None else None,
                coord_feat_2d=None,
                residual_scale_np=self.residual_scale_np,
            )
            nxt_phys = self.normalizer.denorm_state_t(nxt_norm).detach().cpu().numpy()[0]
        if nxt_phys.ndim == 2:
            nxt_phys = nxt_phys[0]
        return np.asarray(nxt_phys, dtype=np.float64)

    def rollout(self, u0: np.ndarray, a_seq: np.ndarray) -> np.ndarray:
        # Prefer the training script rollout to keep every residual/projector detail identical.
        if hasattr(self.train_mod, "rollout_model_np"):
            arr = self.train_mod.rollout_model_np(
                self.model, self.model_type, self.pde_name,
                np.asarray(u0, dtype=np.float32),
                np.asarray(a_seq, dtype=np.float32),
                self.normalizer,
                self.x_norm_1d,
                None,
                self.coord_feat_1d,
                None,
                residual_scale_np=self.residual_scale_np,
                projected=False,
                return_diag=False,
            )
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 3:  # [T,C,N]
                arr = arr[:, 0, :]
            return arr
        u = np.asarray(u0, dtype=np.float64).copy()
        states = [u.copy()]
        for a in a_seq:
            u = self.one_step(u, float(a))
            states.append(u.copy())
        return np.stack(states, axis=0)


def build_model_from_ckpt(ckpt: dict, train_script: Optional[str] = None, x_norm: Optional[np.ndarray] = None):
    """Build a frozen/frozen_model model from a self-describing checkpoint."""
    train_script = train_script or CFG.train_script
    train_mod = load_module_from_path("burgers_frozen_model_training_module", train_script)

    cfg = ckpt["cfg"]
    exp = cfg.get("experiment", {})
    data_meta = cfg.get("data_meta", {})
    pde_name = str(ckpt.get("pde_name", exp.get("pde", "burgers"))).lower()
    model_type = str(ckpt.get("model_type", exp.get("mode", ""))).lower()
    if pde_name != "burgers":
        raise RuntimeError(f"The Burgers decision generator received pde_name={pde_name}.")
    if not bool(exp.get("controlled", True)):
        raise RuntimeError("This data-only script supports controlled Burgers checkpoints only.")

    model_cfg = cfg.get("model_config", {})
    state_shape = tuple(model_cfg.get("state_shape", [1, int(data_meta.get("nx", 128))]))
    coord_dim = int(model_cfg.get("coord_dim", 1 + 2 * len(model_cfg.get("fourier_freqs", [1, 2, 4, 8, 16]))))

    if hasattr(train_mod, "apply_model_training_preset"):
        try:
            train_mod.apply_model_training_preset(pde_name, model_type)
        except Exception as e:
            print(f"[Warn] apply_model_training_preset failed but continuing: {e}")

    model = train_mod.get_model(model_type, pde_name, state_shape, coord_dim)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(DEVICE)

    normalizer = _normalizer_from_ckpt(train_mod, ckpt)
    residual_scale_norm = ckpt.get("residual_scale_norm", None)

    if x_norm is None:
        nx = int(data_meta["nx"])
        L = float(data_meta["L"])
        x = make_grid(nx, L)
        x_norm = 2.0 * (x / L) - 1.0
    surrogate = NC12BurgersSurrogate(train_mod, model, model_type, normalizer, residual_scale_norm, x_norm)
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

# ============================================================
# Load controlled checkpoint
# ============================================================
def controlled_surrogate_one_step(model: nn.Module, model_type: str,
                                  u_t: np.ndarray, a_t: float,
                                  x_norm: np.ndarray) -> np.ndarray:
    if hasattr(model, "one_step"):
        return model.one_step(u_t, float(a_t))
    raise RuntimeError(
        "The compatibility-only in-file surrogate is disabled for frozen runs. "
        "Use build_model_from_ckpt(..., train_script=...) to construct NC12BurgersSurrogate."
    )


def surrogate_rollout(model: nn.Module, model_type: str,
                      u0: np.ndarray, a_seq: np.ndarray, x_norm: np.ndarray) -> np.ndarray:
    if hasattr(model, "rollout"):
        return model.rollout(u0, a_seq)
    u = u0.copy().astype(np.float64)
    states = [u.copy()]
    for a in a_seq:
        u = controlled_surrogate_one_step(model, model_type, u, float(a), x_norm)
        states.append(u.copy())
    return np.stack(states, axis=0)


# ============================================================
# Local linearization of learned controlled surrogate
# ============================================================
def linearize_surrogate(model: nn.Module, model_type: str, x0: np.ndarray, u0: float, x_norm: np.ndarray,
                        eps_x: float, eps_u: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x0.shape[0]
    f0 = controlled_surrogate_one_step(model, model_type, x0, u0, x_norm)

    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps_x
        xm[i] -= eps_x
        fp = controlled_surrogate_one_step(model, model_type, xp, u0, x_norm)
        fm = controlled_surrogate_one_step(model, model_type, xm, u0, x_norm)
        A[:, i] = (fp - fm) / (2.0 * eps_x)

    fp_u = controlled_surrogate_one_step(model, model_type, x0, u0 + eps_u, x_norm)
    fm_u = controlled_surrogate_one_step(model, model_type, x0, u0 - eps_u, x_norm)
    B = ((fp_u - fm_u) / (2.0 * eps_u)).reshape(-1, 1)

    c = f0 - A @ x0 - B[:, 0] * u0
    return A, B, c


# ============================================================
# Hard / OOD IC selection
# ============================================================
def make_ood_variant(u0: np.ndarray, x: np.ndarray, cfg: CFG, rng: np.random.Generator) -> np.ndarray:
    phase = rng.uniform(0.0, 2.0 * np.pi)
    hf = np.sin(cfg.ood_hf_mode * x + phase)
    hf = hf / (np.max(np.abs(hf)) + 1e-12)
    u = cfg.ood_amp_scale * u0 + cfg.ood_hf_amp * hf
    return u.astype(np.float64)


def load_initial_conditions_from_dataset(dataset_path: str,
                                         ckpt_cfg: dict,
                                         n_eval_ic: int,
                                         x_nom: np.ndarray,
                                         dx_nom: float,
                                         cfg: CFG) -> Tuple[np.ndarray, List[str], np.ndarray]:
    data = np.load(dataset_path, allow_pickle=True)
    state = data["state"]
    split_info = ckpt_cfg.get("split_info", {})
    if "test_ids" in split_info and len(split_info["test_ids"]) > 0:
        test_ids = np.array(split_info["test_ids"], dtype=int)
    else:
        J = state.shape[0]
        n_train = int(round(0.7 * J))
        n_val = int(round(0.1 * J))
        test_ids = np.arange(n_train + n_val, J)

    u0_pool = state[test_ids, 0].astype(np.float64)
    z0_pool = np.asarray([initial_risk_z(u, dx_nom) for u in u0_pool], dtype=np.float64)

    order = np.argsort(-z0_pool)
    pool_size = min(len(order), max(n_eval_ic * cfg.candidate_pool_mult, n_eval_ic))
    top_idx = order[:pool_size]
    u0_top = u0_pool[top_idx]
    z0_top = z0_pool[top_idx]

    rng = np.random.default_rng(cfg.seed)

    if cfg.ic_mode == "random":
        ids = rng.choice(len(u0_pool), size=min(n_eval_ic, len(u0_pool)), replace=False)
        return u0_pool[ids], ["test_random"] * len(ids), np.asarray([initial_risk_z(u, dx_nom) for u in u0_pool[ids]])

    if cfg.ic_mode == "high_gradient":
        take = min(n_eval_ic, len(u0_top))
        return u0_top[:take], ["high_grad"] * take, z0_top[:take]

    if cfg.ic_mode == "ood_amplified":
        take = min(n_eval_ic, len(u0_top))
        out = [make_ood_variant(u0_top[i], x_nom, cfg, rng) for i in range(take)]
        out = np.stack(out, axis=0)
        z0 = np.asarray([initial_risk_z(u, dx_nom) for u in out], dtype=np.float64)
        return out, ["ood_amp"] * take, z0

    if cfg.ic_mode == "challenge_conditions":
        n_half = max(1, n_eval_ic // 2)
        take_in = min(n_half, len(u0_top))
        take_ood = min(n_eval_ic - take_in, len(u0_top) - take_in)

        out_list: List[np.ndarray] = []
        tags: List[str] = []

        for i in range(take_in):
            out_list.append(u0_top[i].copy())
            tags.append("high_grad")

        for j in range(take_ood):
            base = u0_top[take_in + j]
            out_list.append(make_ood_variant(base, x_nom, cfg, rng))
            tags.append("ood_amp")

        out = np.stack(out_list, axis=0)
        z0 = np.asarray([initial_risk_z(u, dx_nom) for u in out], dtype=np.float64)
        order2 = np.argsort(-z0)
        out = out[order2]
        z0 = z0[order2]
        tags = [tags[i] for i in order2]
        return out[:n_eval_ic], tags[:n_eval_ic], z0[:n_eval_ic]

    raise ValueError(f"Unknown ic_mode={cfg.ic_mode}")


# ============================================================
# Boundary-seeking MPC on local learned surrogate
# ============================================================
def build_boundary_seeking_mpc(A: np.ndarray, B: np.ndarray, c: np.ndarray,
                               D1: np.ndarray, kappa: np.ndarray, cfg: CFG):
    """
    Performance-seeking controller under nominal surrogate safety certificate:
        maximize actuator-region response
        s.t.   surrogate gradient risk <= z_limit_nom
    """
    nx = A.shape[0]
    H = cfg.H

    X = ca.SX.sym("X", nx, H + 1)
    U = ca.SX.sym("U", H)
    x0 = ca.SX.sym("x0", nx)
    u_prev = ca.SX.sym("u_prev")
    z_limit_nom = ca.SX.sym("z_limit_nom")

    A_dm = ca.DM(A)
    B_dm = ca.DM(B)
    c_dm = ca.DM(c.reshape(-1, 1))
    D1_dm = ca.DM(D1)
    k_dm = ca.DM(kappa.reshape(-1, 1))

    g_eq = []
    g_ineq = []
    g_eq.append(X[:, 0] - x0)

    J = 0.0
    p = cfg.grad_p
    eps = cfg.grad_eps

    for t in range(H):
        xt = X[:, t]
        ut = U[t]
        x_next = A_dm @ xt + B_dm * ut + c_dm
        g_eq.append(X[:, t + 1] - x_next)

        # surrogate safety constraint on next state
        grad_next = D1_dm @ X[:, t + 1]
        grad_abs = ca.sqrt(grad_next**2 + eps**2)
        z_next = ca.power(ca.sum1(ca.power(grad_abs, p)), 1.0 / p)
        g_ineq.append(z_next)

        # performance reward: maximize actuator-region response magnitude
        perf_next = ca.dot(k_dm, X[:, t + 1]) / nx
        J += -cfg.w_perf_stage * perf_next**2

        # very weak regularization so optimizer can really use control
        J += cfg.w_energy_reg * ca.sumsqr(X[:, t + 1]) / nx
        J += cfg.w_ctrl * ut**2
        if t == 0:
            J += cfg.w_du * (ut - u_prev)**2
        else:
            J += cfg.w_du * (ut - U[t - 1])**2

    perf_terminal = ca.dot(k_dm, X[:, H]) / nx
    J += -cfg.w_perf_terminal * perf_terminal**2

    w = ca.vertcat(ca.reshape(X, -1, 1), U)
    gcat = ca.vertcat(*(g_eq + g_ineq))
    pcat = ca.vertcat(x0, u_prev, z_limit_nom)

    nlp = {"x": w, "f": J, "g": gcat, "p": pcat}
    opts = {
        "ipopt.print_level": cfg.ipopt_print_level,
        "print_time": False,
        "ipopt.max_iter": cfg.max_iter,
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    nX = nx * (H + 1)
    lbx = np.full(nX + H, -np.inf)
    ubx = np.full(nX + H, np.inf)
    lbx[nX:] = cfg.umin
    ubx[nX:] = cfg.umax

    # equality part
    n_eq = nx * (H + 1)
    # inequality part: H stage safety values
    n_ineq = H
    lbg = np.concatenate([np.zeros(n_eq), -np.inf * np.ones(n_ineq)])
    ubg = np.concatenate([np.zeros(n_eq), np.zeros(n_ineq)])  # placeholder; will be shifted by z_limit_nom through g - zlim <= 0 reform below?

    return solver, lbx, ubx, lbg, ubg, nX


def build_boundary_seeking_mpc_shifted(A: np.ndarray, B: np.ndarray, c: np.ndarray,
                                       D1: np.ndarray, kappa: np.ndarray, cfg: CFG):
    """
    Same as above, but inequality stored as z_next - z_limit_nom <= 0
    so ubg = 0 is fixed.
    """
    nx = A.shape[0]
    H = cfg.H

    X = ca.SX.sym("X", nx, H + 1)
    U = ca.SX.sym("U", H)
    x0 = ca.SX.sym("x0", nx)
    u_prev = ca.SX.sym("u_prev")
    z_limit_nom = ca.SX.sym("z_limit_nom")

    A_dm = ca.DM(A)
    B_dm = ca.DM(B)
    c_dm = ca.DM(c.reshape(-1, 1))
    D1_dm = ca.DM(D1)
    k_dm = ca.DM(kappa.reshape(-1, 1))

    g_eq = []
    g_ineq = []
    g_eq.append(X[:, 0] - x0)

    J = 0.0
    p = cfg.grad_p
    eps = cfg.grad_eps

    for t in range(H):
        xt = X[:, t]
        ut = U[t]
        x_next = A_dm @ xt + B_dm * ut + c_dm
        g_eq.append(X[:, t + 1] - x_next)

        grad_next = D1_dm @ X[:, t + 1]
        grad_abs = ca.sqrt(grad_next**2 + eps**2)
        z_next = ca.power(ca.sum1(ca.power(grad_abs, p)), 1.0 / p)
        g_ineq.append(z_next - z_limit_nom)

        perf_next = ca.dot(k_dm, X[:, t + 1]) / nx
        J += -cfg.w_perf_stage * perf_next**2
        J += cfg.w_energy_reg * ca.sumsqr(X[:, t + 1]) / nx
        J += cfg.w_ctrl * ut**2
        if t == 0:
            J += cfg.w_du * (ut - u_prev)**2
        else:
            J += cfg.w_du * (ut - U[t - 1])**2

    perf_terminal = ca.dot(k_dm, X[:, H]) / nx
    J += -cfg.w_perf_terminal * perf_terminal**2

    w = ca.vertcat(ca.reshape(X, -1, 1), U)
    gcat = ca.vertcat(*(g_eq + g_ineq))
    pcat = ca.vertcat(x0, u_prev, z_limit_nom)

    nlp = {"x": w, "f": J, "g": gcat, "p": pcat}
    opts = {
        "ipopt.print_level": cfg.ipopt_print_level,
        "print_time": False,
        "ipopt.max_iter": cfg.max_iter,
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    nX = nx * (H + 1)
    lbx = np.full(nX + H, -np.inf)
    ubx = np.full(nX + H, np.inf)
    lbx[nX:] = cfg.umin
    ubx[nX:] = cfg.umax

    n_eq = nx * (H + 1)
    n_ineq = H
    lbg = np.concatenate([np.zeros(n_eq), -np.inf * np.ones(n_ineq)])
    ubg = np.concatenate([np.zeros(n_eq), np.zeros(n_ineq)])

    return solver, lbx, ubx, lbg, ubg, nX


def solve_open_loop_mpc(x0_nom: np.ndarray, u_prev: float, z_limit_nom: float,
                        solver, lbx, ubx, lbg, ubg, nX: int, cfg: CFG):
    nx = x0_nom.shape[0]
    H = cfg.H

    X0 = np.tile(x0_nom.reshape(-1, 1), (1, H + 1)).reshape(-1)
    U0 = np.zeros(H)
    w0 = np.concatenate([X0, U0])

    p = np.concatenate([x0_nom, np.array([u_prev, z_limit_nom])])
    sol = solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)

    w_opt = np.array(sol["x"]).reshape(-1)
    X_opt = w_opt[:nX].reshape(nx, H + 1).T.copy()
    U_opt = w_opt[nX:].copy()
    J_opt = float(np.array(sol["f"]).reshape(()))
    return X_opt, U_opt, J_opt


# ============================================================
# Plot-ready data export utilities
# ============================================================
import csv
import shutil
from datetime import datetime
from typing import Dict, Any


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


def risk_inf_series(states: np.ndarray, dx: float) -> np.ndarray:
    return np.asarray([initial_risk_z(u, dx) for u in states], dtype=np.float64)


def response_series_1d(states: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    return np.asarray([terminal_response(u, kappa) for u in states[1:]], dtype=np.float64)


def interp_true_states_to_nom(states_true: np.ndarray, x_true: np.ndarray, x_nom: np.ndarray) -> np.ndarray:
    """Return true rollout interpolated onto the nominal grid for plot-ready field comparison."""
    return np.stack([periodic_interp_1d(u, x_true, x_nom) for u in states_true], axis=0).astype(np.float64)


def make_selected_case_maps_1d(states_nom: np.ndarray,
                               states_true_on_nom: np.ndarray,
                               dx_nom: float,
                               n_times: int = 4):
    """
    Save 1D plot-ready fields in a 2D-compatible shape [K, 1, N].
    The stored arrays support the manuscript field and risk-trace visualizations.
    """
    ids = np.linspace(0, len(states_nom) - 1, n_times, dtype=int)
    u_sur = []
    u_hf = []
    err = []
    grad_sur = []
    grad_hf = []
    for k in ids:
        us = np.asarray(states_nom[k], dtype=np.float64)
        uh = np.asarray(states_true_on_nom[k], dtype=np.float64)
        ux_s = (np.roll(us, -1) - np.roll(us, 1)) / (2.0 * dx_nom)
        ux_h = (np.roll(uh, -1) - np.roll(uh, 1)) / (2.0 * dx_nom)
        u_sur.append(us[None, :].astype(np.float32))
        u_hf.append(uh[None, :].astype(np.float32))
        err.append(np.abs(uh - us)[None, :].astype(np.float32))
        grad_sur.append(np.abs(ux_s)[None, :].astype(np.float32))
        grad_hf.append(np.abs(ux_h)[None, :].astype(np.float32))
    return (
        ids,
        np.stack(u_sur, axis=0),
        np.stack(u_hf, axis=0),
        np.stack(err, axis=0),
        np.stack(grad_sur, axis=0),
        np.stack(grad_hf, axis=0),
    )


def update_top_cases(top_cases: List[Dict[str, Any]], pack: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
    top_cases.append(pack)
    # Sort by positive risk gap; if all are negative, this still keeps the largest gap.
    top_cases = sorted(top_cases, key=lambda d: float(d["gap"]), reverse=True)
    return top_cases[:k]


def cfg_to_json_dict(cfg: CFG) -> Dict[str, Any]:
    d = asdict(cfg)
    # Include dynamic attributes added after checkpoint loading.
    for key in ["nx_nom", "L", "nu", "dt_nom", "ctrl_center", "ctrl_sigma"]:
        if hasattr(cfg, key):
            d[key] = getattr(cfg, key)
    return json_safe(d)


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
            # Use [T,1,N] arrays so the generic plotting script can handle Burgers as a thin field.
            states_sur=np.asarray(c["states_sur_plot"], dtype=np.float32),
            states_hf=np.asarray(c["states_hf_plot"], dtype=np.float32),
            # Also retain raw 1D arrays and true-grid arrays for custom Burgers plots.
            states_sur_1d=np.asarray(c["states_sur"], dtype=np.float32),
            states_hf_1d_on_nom=np.asarray(c["states_hf_on_nom"], dtype=np.float32),
            states_hf_true_grid=np.asarray(c["states_hf_true"], dtype=np.float32),
            x_nom=np.asarray(c["x_nom"], dtype=np.float32),
            x_true=np.asarray(c["x_true"], dtype=np.float32),
            z_sur_ts=np.asarray(c["z_sur_ts"], dtype=np.float64),
            z_hf_ts=np.asarray(c["z_hf_ts"], dtype=np.float64),
            J_sur_ts=np.asarray(c["J_sur_ts"], dtype=np.float64),
            J_hf_ts=np.asarray(c["J_hf_ts"], dtype=np.float64),
            time=np.arange(len(c["z_sur_ts"]), dtype=np.float64) * float(cfg.dt_nom),
            time_state=np.arange(len(c["z_sur_ts"]), dtype=np.float64) * float(cfg.dt_nom),
            time_control=np.arange(len(c["U_opt"]), dtype=np.float64) * float(cfg.dt_nom),
            selected_time_ids=np.asarray(c["selected_time_ids"], dtype=np.int64),
            u_sur_selected=np.asarray(c["u_sur_selected"], dtype=np.float32),
            u_hf_selected=np.asarray(c["u_hf_selected"], dtype=np.float32),
            err_selected=np.asarray(c["err_selected"], dtype=np.float32),
            grad_sur_selected=np.asarray(c["grad_sur_selected"], dtype=np.float32),
            grad_hf_selected=np.asarray(c["grad_hf_selected"], dtype=np.float32),
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
# Main: compute only, save plot-ready data, no figure generation
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_script", type=str, default=CFG.train_script,
                        help="Path to train_discovery_surrogates.py")
    parser.add_argument("--ckpt_path", type=str, default=CFG.ckpt_path)
    parser.add_argument("--dataset_path", type=str, default=CFG.dataset_path)
    parser.add_argument("--outdir", type=str, default=CFG.outdir,
                        help="Output root. By default, a timestamped run subdirectory is created inside it.")
    parser.add_argument("--H", type=int, default=CFG.H)
    parser.add_argument("--n_eval_ic", type=int, default=CFG.n_eval_ic)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--nx_true", type=int, default=CFG.nx_true)
    parser.add_argument("--dt_true", type=float, default=CFG.dt_true)
    parser.add_argument("--ic_mode", type=str, default=CFG.ic_mode,
                        choices=["random", "high_gradient", "ood_amplified", "challenge_conditions"])
    parser.add_argument("--z_limit_scale", type=float, default=CFG.z_limit_scale)
    parser.add_argument("--boundary_mode", choices=("initial_risk",), default=CFG.boundary_mode)
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
        nx_true=args.nx_true,
        dt_true=args.dt_true,
        ic_mode=args.ic_mode,
        z_limit_scale=args.z_limit_scale,
        boundary_mode=args.boundary_mode,
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    model, model_type, data_meta = build_model_from_ckpt(ckpt, train_script=cfg.train_script)
    ckpt_cfg = ckpt["cfg"]

    pde = str(ckpt.get("pde_name", ckpt_cfg.get("experiment", {}).get("pde", "burgers")))
    if pde != "burgers" or not ckpt_cfg.get("experiment", {}).get("controlled", True):
        raise RuntimeError("This data-only script currently supports controlled Burgers checkpoints only.")

    run_id, run_dir, prefix = build_run_paths(args.outdir, pde, model_type, cfg.ic_mode, timestamp, bool(args.flat_outdir))
    cfg.outdir = run_dir
    ensure_dir(cfg.outdir)

    # grids from checkpoint/meta
    nx_nom = int(data_meta["nx"])
    cfg.nx_nom = nx_nom
    cfg.L = float(data_meta["L"])
    cfg.nu = float(data_meta["nu"])
    cfg.dt_nom = float(data_meta["dt"])
    cfg.umin = float(data_meta["u_min"])
    cfg.umax = float(data_meta["u_max"])
    cfg.ctrl_center = float(data_meta["ctrl_center"])
    cfg.ctrl_sigma = float(data_meta["ctrl_sigma"])

    x_nom = make_grid(cfg.nx_nom, cfg.L)
    x_true = make_grid(cfg.nx_true, cfg.L)
    D1_nom, D2_nom, dx_nom = build_periodic_D1_D2(cfg.nx_nom, cfg.L)
    dx_true = cfg.L / cfg.nx_true

    k_nom = control_profile(x_nom, cfg.ctrl_center, cfg.ctrl_sigma, cfg.L)
    k_true = control_profile(x_true, cfg.ctrl_center, cfg.ctrl_sigma, cfg.L)
    x_norm = 2.0 * (x_nom / cfg.L) - 1.0

    # ---------------- hard / OOD IC selection ----------------
    u0s_nom, ic_tags, z0s = load_initial_conditions_from_dataset(
        cfg.dataset_path, ckpt_cfg, cfg.n_eval_ic, x_nom, dx_nom, cfg
    )

    z_sur_list, z_hf_list, z_limit_list, gap_list = [], [], [], []
    J_sur_list, J_hf_list = [], []
    nominal_cost_list = []
    surrogate_excess_list, hf_excess_list = [], []
    margin_sur_list, margin_hf_list = [], []
    per_ic_rows: List[Dict[str, Any]] = []
    U_all, alpha_all = [], []
    z_sur_ts_all, z_hf_ts_all = [], []
    J_sur_ts_all, J_hf_ts_all = [], []
    top_cases: List[Dict[str, Any]] = []

    t_start = datetime.now()

    for i, u0_nom in enumerate(u0s_nom):
        z0 = initial_risk_z(u0_nom, dx_nom)
        z_limit_nom = cfg.z_limit_scale * z0 + cfg.z_limit_margin
        print(f"[Run:{run_id}] IC {i+1}/{len(u0s_nom)} | tag={ic_tags[i]} | Z0={z0:.6f} | z_limit_nom={z_limit_nom:.6f}")

        # local linearization of learned surrogate around (u0_nom, u_prev)
        A, B, c = linearize_surrogate(
            model=model,
            model_type=model_type,
            x0=u0_nom,
            u0=cfg.u_prev,
            x_norm=x_norm,
            eps_x=cfg.eps_x,
            eps_u=cfg.eps_u,
        )

        solver, lbx, ubx, lbg, ubg, nX = build_boundary_seeking_mpc_shifted(A, B, c, D1_nom, k_nom, cfg)
        X_loc_opt, U_opt, J_opt = solve_open_loop_mpc(
            x0_nom=u0_nom,
            u_prev=cfg.u_prev,
            z_limit_nom=z_limit_nom,
            solver=solver,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
            nX=nX,
            cfg=cfg,
        )

        # learned surrogate rollout under same optimized controls
        states_sur = surrogate_rollout(
            model=model,
            model_type=model_type,
            u0=u0_nom,
            a_seq=U_opt,
            x_norm=x_norm,
        )
        z_sur = risk_Z_inf(states_sur, dx_nom)
        J_sur_final = terminal_response(states_sur[-1], k_nom)

        # true env rollout under same controls
        u0_true = periodic_interp_1d(u0_nom, x_nom, x_true)
        states_true = rollout_true(
            u0_true=u0_true,
            a_seq=U_opt,
            x_true=x_true,
            k_true=k_true,
            nu=cfg.nu,
            dt_nom=cfg.dt_nom,
            dt_true=cfg.dt_true,
        )
        z_hf = risk_Z_inf(states_true, dx_true)
        J_hf_final = terminal_response(states_true[-1], k_true)

        states_hf_on_nom = interp_true_states_to_nom(states_true, x_true, x_nom)

        z_sur_ts = risk_inf_series(states_sur, dx_nom)
        z_hf_ts = risk_inf_series(states_true, dx_true)
        J_sur_ts = response_series_1d(states_sur, k_nom)
        J_hf_ts = response_series_1d(states_true, k_true)

        gap = float(z_hf - z_sur)
        surrogate_excess = float(z_sur - z_limit_nom)
        hf_excess = float(z_hf - z_limit_nom)
        margin_sur = float(z_limit_nom - z_sur)
        margin_hf = float(z_limit_nom - z_hf)

        z_sur_list.append(float(z_sur))
        z_hf_list.append(float(z_hf))
        z_limit_list.append(float(z_limit_nom))
        gap_list.append(gap)
        J_sur_list.append(float(J_sur_final))
        J_hf_list.append(float(J_hf_final))
        nominal_cost_list.append(float(J_opt))
        surrogate_excess_list.append(surrogate_excess)
        hf_excess_list.append(hf_excess)
        margin_sur_list.append(margin_sur)
        margin_hf_list.append(margin_hf)
        U_all.append(U_opt.astype(np.float32))
        # Burgers directly optimizes the full control sequence; alpha is retained for a common archive schema.
        alpha_all.append(U_opt.astype(np.float32))
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
            "tag": ic_tags[i],
            "z0": float(z0),
            "z_sur": float(z_sur),
            "z_hf": float(z_hf),
            "gap": gap,
            "z_limit": float(z_limit_nom),
            "margin_sur": margin_sur,
            "margin_hf": margin_hf,
            "surrogate_excess": surrogate_excess,
            "hf_excess": hf_excess,
            "nominal_excess": surrogate_excess,
            "true_excess": hf_excess,
            "J_sur": float(J_sur_final),
            "J_hf": float(J_hf_final),
            "J_gap": float(J_hf_final - J_sur_final),
            "nominal_cost": float(J_opt),
            "underestimated": int(gap > 0.0),
            "surrogate_safe": int(z_sur <= z_limit_nom),
            "hf_safe": int(z_hf <= z_limit_nom),
            "surrogate_safe_hf_unsafe": int((z_sur <= z_limit_nom) and (z_hf > z_limit_nom)),
            "U_mean": float(np.mean(U_opt)),
            "U_std": float(np.std(U_opt)),
            "U_min": float(np.min(U_opt)),
            "U_max": float(np.max(U_opt)),
            "U_tv": float(np.sum(np.abs(np.diff(U_opt)))) if len(U_opt) > 1 else 0.0,
        }
        per_ic_rows.append(row)

        selected_ids, u_sur_sel, u_hf_sel, err_sel, grad_sur_sel, grad_hf_sel = make_selected_case_maps_1d(
            states_sur, states_hf_on_nom, dx_nom, n_times=4
        )
        top_cases = update_top_cases(top_cases, {
            "ic_idx": i,
            "tag": ic_tags[i],
            "pde": pde,
            "model_type": model_type,
            "ic_mode": cfg.ic_mode,
            "timestamp": timestamp,
            "z0": float(z0),
            "states_sur": states_sur.copy(),
            "states_hf_true": states_true.copy(),
            "states_hf_on_nom": states_hf_on_nom.copy(),
            "states_sur_plot": states_sur[:, None, :].copy(),
            "states_hf_plot": states_hf_on_nom[:, None, :].copy(),
            "x_nom": x_nom.copy(),
            "x_true": x_true.copy(),
            "U_opt": U_opt.copy(),
            "alpha_opt": U_opt.copy(),
            "z_sur": float(z_sur),
            "z_hf": float(z_hf),
            "gap": gap,
            "z_limit": float(z_limit_nom),
            "J_sur": float(J_sur_final),
            "J_hf": float(J_hf_final),
            "z_sur_ts": z_sur_ts.copy(),
            "z_hf_ts": z_hf_ts.copy(),
            "J_sur_ts": J_sur_ts.copy(),
            "J_hf_ts": J_hf_ts.copy(),
            "selected_time_ids": selected_ids,
            "u_sur_selected": u_sur_sel,
            "u_hf_selected": u_hf_sel,
            "err_selected": err_sel,
            "grad_sur_selected": grad_sur_sel,
            "grad_hf_selected": grad_hf_sel,
        }, k=int(args.top_k_cases))

    z_sur_arr = np.asarray(z_sur_list, dtype=np.float64)
    z_hf_arr = np.asarray(z_hf_list, dtype=np.float64)
    z_limit_arr = np.asarray(z_limit_list, dtype=np.float64)
    gap_arr = np.asarray(gap_list, dtype=np.float64)
    J_sur_arr = np.asarray(J_sur_list, dtype=np.float64)
    J_hf_arr = np.asarray(J_hf_list, dtype=np.float64)
    nominal_cost_arr = np.asarray(nominal_cost_list, dtype=np.float64)
    surrogate_excess_arr = np.asarray(surrogate_excess_list, dtype=np.float64)
    hf_excess_arr = np.asarray(hf_excess_list, dtype=np.float64)
    margin_sur_arr = np.asarray(margin_sur_list, dtype=np.float64)
    margin_hf_arr = np.asarray(margin_hf_list, dtype=np.float64)

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
        "nx_nom": int(cfg.nx_nom),
        "nx_true": int(cfg.nx_true),
        "dt_nom": float(cfg.dt_nom),
        "dt_true": float(cfg.dt_true),
        "elapsed_sec": float(elapsed_sec),
        "mean_initial_risk": float(np.mean(z0s)),
        "mean_surrogate_risk": float(np.mean(z_sur_arr)),
        "mean_hf_risk": float(np.mean(z_hf_arr)),
        "mean_gap_hf_minus_surrogate": float(np.mean(gap_arr)),
        "median_gap_hf_minus_surrogate": float(np.median(gap_arr)),
        "q90_gap": q(gap_arr, 0.90),
        "q95_gap": q(gap_arr, 0.95),
        "cvar95_positive_gap": cvar95_gap,
        "fraction_underestimated": frac(gap_arr > 0.0),
        "pearson_r_surrogate_hf_risk": float(np.corrcoef(z_sur_arr, z_hf_arr)[0, 1]) if len(z_sur_arr) >= 2 else float("nan"),
        "surrogate_safe_rate": frac(z_sur_arr <= z_limit_arr),
        "hf_safe_rate": frac(z_hf_arr <= z_limit_arr),
        "surrogate_safe_hf_unsafe_rate": frac((z_sur_arr <= z_limit_arr) & (z_hf_arr > z_limit_arr)),
        "mean_surrogate_violation_excess": float(np.mean(surrogate_excess_arr)),
        "mean_hf_violation_excess": float(np.mean(hf_excess_arr)),
        "mean_surrogate_perf": float(np.mean(J_sur_arr)),
        "mean_hf_perf": float(np.mean(J_hf_arr)),
        "mean_perf_gap_hf_minus_surrogate": float(np.mean(J_hf_arr - J_sur_arr)),
        "worst_gap": float(np.max(gap_arr)),
        "best_gap": float(np.min(gap_arr)),
        "worst_hf_excess": float(np.max(hf_excess_arr)),
        "best_hf_excess": float(np.min(hf_excess_arr)),
        "cfg": cfg_to_json_dict(cfg),
        "data_meta": json_safe(data_meta),
        "frozen_checkpoint_metadata": json_safe(getattr(model, "ckpt_meta", {})),
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
        tags=np.asarray(ic_tags, dtype=object),
        initial_states=np.asarray(u0s_nom, dtype=np.float32),
        z0=np.asarray(z0s, dtype=np.float32),
        z_sur=z_sur_arr.astype(np.float32),
        z_pred=z_sur_arr.astype(np.float32),
        z_hf=z_hf_arr.astype(np.float32),
        z_true=z_hf_arr.astype(np.float32),
        gap=gap_arr.astype(np.float32),
        J_sur=J_sur_arr.astype(np.float32),
        J_hf=J_hf_arr.astype(np.float32),
        J_pred=J_sur_arr.astype(np.float32),
        J_true=J_hf_arr.astype(np.float32),
        z_limit=z_limit_arr.astype(np.float32),
        z_limit_nom=z_limit_arr.astype(np.float32),
        margin_sur=margin_sur_arr.astype(np.float32),
        margin_hf=margin_hf_arr.astype(np.float32),
        surrogate_excess=surrogate_excess_arr.astype(np.float32),
        hf_excess=hf_excess_arr.astype(np.float32),
        nominal_excess=surrogate_excess_arr.astype(np.float32),
        true_excess=hf_excess_arr.astype(np.float32),
        nominal_cost=nominal_cost_arr.astype(np.float32),
        U_all=np.asarray(U_all, dtype=np.float32),
        alpha_all=np.asarray(alpha_all, dtype=np.float32),
        z_sur_ts_all=np.asarray(z_sur_ts_all, dtype=np.float32),
        z_hf_ts_all=np.asarray(z_hf_ts_all, dtype=np.float32),
        J_sur_ts_all=np.asarray(J_sur_ts_all, dtype=np.float32),
        J_hf_ts_all=np.asarray(J_hf_ts_all, dtype=np.float32),
        time_state=np.arange(cfg.H + 1, dtype=np.float32) * float(cfg.dt_nom),
        time_control=np.arange(cfg.H, dtype=np.float32) * float(cfg.dt_nom),
        x_nom=np.asarray(x_nom, dtype=np.float32),
        x_true=np.asarray(x_true, dtype=np.float32),
        config=np.array([json.dumps(cfg_to_json_dict(cfg))], dtype=object),
        summary=np.array([json.dumps(json_safe(summary))], dtype=object),
    )

    # Convenience copies with stable names for plotting scripts.
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
        "train_script": cfg.train_script,
        "frozen_checkpoint_metadata": json_safe(getattr(model, "ckpt_meta", {})),
        "note": "Burgers cases store 1D rollouts and [T,1,N] state arrays for common figure generation.",
    }
    write_json(manifest_path, manifest)

    print("[OK] Burgers decision archive saved")
    print("[run_dir]", cfg.outdir)
    print("[manifest]", manifest_path)
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Shared surrogate and reference-PDE evaluation utilities.

The module loads each frozen surrogate--PDE pair, reconstructs the challenge
conditions, evaluates optimized control sequences, and assigns the four decision
regimes used in the manuscript. It is imported by the operator-defect,
critical-event, mismatch-detection, and recoverability analyses.
"""
from __future__ import annotations

import os
import sys
import csv
import json
import math
import time
import glob
import argparse
import inspect
import importlib.util
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, *args, **kwargs):
        return x if x is not None else []


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PDE_LIST = ["burgers", "grayscott", "kolmogorov"]
MODEL_LIST = ["deeponet", "fno", "pino", "pinn"]
REGIME_NAMES = ["safe_agreement", "false_safe_optimism", "conservative_rejection", "unsafe_agreement"]
MODE_NAMES = ["identity_minimal", "safety_repair", "performance_recovery", "robust_risk_reduction"]
REGIME_TO_MODE = {
    "safe_agreement": "identity_minimal",
    "false_safe_optimism": "safety_repair",
    "conservative_rejection": "performance_recovery",
    "unsafe_agreement": "robust_risk_reduction",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)


def json_safe(x: Any):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_module_from_path(module_name: str, file_path: str):
    file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_pairs(spec: str) -> List[Tuple[str, str]]:
    out = []
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Pair must be pde:model, got {token}")
        pde, model = token.split(":", 1)
        pde, model = pde.strip().lower(), model.strip().lower()
        if pde not in PDE_LIST:
            raise ValueError(f"Unknown PDE {pde}")
        if model not in MODEL_LIST:
            raise ValueError(f"Unknown model {model}")
        out.append((pde, model))
    return out


def parse_name_list(spec: str, allowed: List[str], name: str) -> List[str]:
    out = []
    for token in (spec or "").split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t not in allowed:
            raise ValueError(f"Unknown {name}: {t}; allowed={allowed}")
        out.append(t)
    return out


def parse_float_grid(spec: str, default: List[float]) -> List[float]:
    vals = []
    for token in (spec or "").split(","):
        token = token.strip()
        if token:
            try:
                vals.append(float(token))
            except Exception:
                pass
    return vals if vals else list(default)


def parse_int_grid(spec: str, default: List[int]) -> List[int]:
    vals = []
    for token in (spec or "").split(","):
        token = token.strip()
        if token:
            try:
                vals.append(max(1, int(token)))
            except Exception:
                pass
    return sorted(set(vals)) if vals else list(default)


def rel_gain(new_value: float, old_value: float) -> float:
    return float((float(new_value) - float(old_value)) / (abs(float(old_value)) + 1e-8))


def rel_loss(new_value: float, old_value: float) -> float:
    return float(max(-rel_gain(new_value, old_value), 0.0))


def build_pairs_from_cfg(cfg) -> List[Tuple[str, str]]:
    """Unified pair interface. --run_suite uses pde_list x model_list; --pairs remains supported."""
    if bool(getattr(cfg, "run_suite", False)):
        pdes = parse_name_list(getattr(cfg, "pde_list", ""), PDE_LIST, "PDE")
        models = parse_name_list(getattr(cfg, "model_list", ""), MODEL_LIST, "model")
        return [(p, m) for p in pdes for m in models]
    if getattr(cfg, "pairs", ""):
        return parse_pairs(cfg.pairs)
    pdes = parse_name_list(getattr(cfg, "pde_list", "burgers,grayscott,kolmogorov"), PDE_LIST, "PDE")
    models = parse_name_list(getattr(cfg, "model_list", "deeponet,fno,pino,pinn"), MODEL_LIST, "model")
    return [(p, m) for p in pdes for m in models]


def pair_key(pde: str, model: str) -> str:
    return f"{pde}:{model}"


def pair_dir_name(pde: str, model: str) -> str:
    return f"{pde}_{model}"


def interp_1d(y: np.ndarray, n_out: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n_in = len(y)
    if n_in == n_out:
        return y.copy()
    if n_in <= 1:
        return np.full(n_out, y[0] if n_in else 0.0, dtype=np.float64)
    xi = np.linspace(0.0, 1.0, n_in)
    xo = np.linspace(0.0, 1.0, n_out)
    return np.interp(xo, xi, y)


def moving_average(u: np.ndarray, k: int = 5) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    k = max(1, int(k))
    if k <= 1:
        return u.copy()
    pad = k // 2
    up = np.pad(u, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=np.float64) / k
    return np.convolve(up, ker, mode="valid")[:len(u)]


def total_variation(u: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(np.diff(u)))) if len(u) > 1 else 0.0


def high_freq_energy_ratio(u: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if len(u) <= 4:
        return 0.0
    f = np.fft.rfft(u - np.mean(u))
    e = np.abs(f) ** 2
    if e.sum() <= 1e-12:
        return 0.0
    cut = max(1, len(e) // 3)
    return float(e[cut:].sum() / e.sum())


def classify_regime(m_s: float, m_hf: float) -> str:
    if m_s >= 0.0 and m_hf >= 0.0:
        return "safe_agreement"
    if m_s >= 0.0 and m_hf < 0.0:
        return "false_safe_optimism"
    if m_s < 0.0 and m_hf >= 0.0:
        return "conservative_rejection"
    return "unsafe_agreement"


def one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float64)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def robust_scale_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(X, axis=0)
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    scale = (q3 - q1) / 1.349
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, np.nanstd(X, axis=0) + 1e-8)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return med.astype(np.float64), scale.astype(np.float64)


def robust_scale_apply(X: np.ndarray, med: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((X - med) / scale).astype(np.float32)


def save_csv(rows: List[Dict[str, Any]], path: str):
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class Config:
    root: str = "./external_data/decision_archives/discovery"
    ckpt_root: str = "./external_data/checkpoints/discovery"
    data_root: str = "./external_data/datasets/discovery"
    train_script: str = "./scripts/train_discovery_surrogates.py"
    burgers_script: str = "./scripts/generate_burgers_decisions.py"
    grayscott_script: str = "./scripts/generate_gray_scott_decisions.py"
    kolmogorov_script: str = "./scripts/generate_kolmogorov_decisions.py"
    outdir: str = "./outputs/analysis/surrogate_pair_evaluation"
    ic_mode: str = "challenge_conditions"
    pairs: str = ""
    train_pairs: str = ""
    test_pairs: str = ""
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    H: int = 80
    canonical_horizon: int = 80
    max_samples_per_pair: int = -1
    candidate_count: int = 16
    random_candidates: int = 6
    epochs: int = 300
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 256
    depth: int = 4
    dropout: float = 0.05
    trust_min: float = 0.02
    trust_max: float = 0.85
    lambda_control_mse: float = 1.0
    lambda_smooth: float = 0.02
    lambda_identity: float = 0.3
    safety_margin_target: float = 0.02
    distance_penalty: float = 0.08
    performance_weight: float = 1.0
    safety_weight: float = 10.0
    candidate_noise_scale: float = 0.08
    candidate_smooth_k: int = 7
    dpi: int = 360
    no_train: int = 0
    rebuild_cache: int = 0
    # suite interface
    run_suite: bool = False
    pde_list: str = "burgers,grayscott,kolmogorov"
    model_list: str = "deeponet,fno,pino,pinn"
    # generalization split
    split_mode: str = "ic"          # ic | pair | leave_pde | leave_model
    val_fraction: float = 0.15
    test_fraction: float = 0.20
    holdout_pde: str = ""
    holdout_model: str = ""
    early_stopping_patience: int = 40
    min_delta: float = 1e-5
    # projection objective and evaluation
    margin_cap: float = 0.08
    j_tolerance: float = 1e-4
    no_harm_weight: float = 8.0
    teacher_identity_distance_weight: float = 20.0
    teacher_distance_weight: float = 2.0
    # Report performance by decision regime rather than averaging incompatible outcomes.
    relative_j_tolerance: float = 1e-3
    performance_loss_budget_safety: float = 0.05
    performance_loss_budget_unsafe: float = 0.10
    performance_gain_target: float = 1e-3
    candidate_alpha_grid: str = "0,0.05,0.10,0.20,0.35,0.50,0.70,0.90,1.0"
    candidate_smooth_grid: str = "1,3,5,9,15"
    force_identity_for_safe_agreement: int = 1
    reevaluate_baseline: int = 1
    # model architecture
    model_arch: str = "mlp"         # mlp | attention
    nhead: int = 4
    transformer_layers: int = 2


# -----------------------------------------------------------------------------
# Decision-archive loading
# -----------------------------------------------------------------------------
def resolve_decision_archive(root: str, pde: str, model: str, ic_mode: str) -> str:
    direct = os.path.join(root, pde, model, ic_mode)
    if os.path.isdir(direct):
        return direct
    hits = []
    for r, dirs, files in os.walk(root):
        if "decision_data.npz" in files or "plot_data.npz" in files:
            text = r.lower()
            if pde in text and model in text and ic_mode in text:
                hits.append(r)
    if not hits:
        raise FileNotFoundError(f"Cannot find a decision archive for {pde}/{model}/{ic_mode} under {root}")
    hits.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return hits[0]


def load_decision_archive(root: str, pde: str, model: str, ic_mode: str) -> Dict[str, Any]:
    d = resolve_decision_archive(root, pde, model, ic_mode)
    npz_path = os.path.join(d, "decision_data.npz")
    if not os.path.exists(npz_path):
        alt = glob.glob(os.path.join(d, "*_plot_data.npz")) + glob.glob(os.path.join(d, "plot_data.npz"))
        if not alt:
            raise FileNotFoundError(f"No decision-array data in {d}")
        npz_path = alt[0]
    arr = np.load(npz_path, allow_pickle=True)
    csv_path = os.path.join(d, "decision_summary.csv")
    if not os.path.exists(csv_path):
        alt = glob.glob(os.path.join(d, "*_per_ic.csv"))
        csv_path = alt[0] if alt else ""
    df = pd.read_csv(csv_path) if csv_path else None
    return {"dir": d, "npz_path": npz_path, "csv_path": csv_path, "npz": arr, "df": df}


def resolve_checkpoint(ckpt_root: str, pde: str, model: str) -> str:
    public_pde = "gray_scott" if pde == "grayscott" else pde
    public_candidates = [
        os.path.join(ckpt_root, "discovery", public_pde, f"{model}.pt"),
        os.path.join(ckpt_root, public_pde, f"{model}.pt"),
        os.path.join(ckpt_root, public_pde, model, "best_checkpoint.pt"),
    ]
    for candidate in public_candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Cannot find checkpoint for {pde}/{model} under {ckpt_root}. "
        "Use the public checkpoints/discovery/<pde>/<model>.pt layout."
    )


def resolve_dataset(data_root: str, pde: str) -> str:
    public_pde = "gray_scott" if pde == "grayscott" else pde
    cands = [
        os.path.join(data_root, "discovery", f"{public_pde}_controlled_trajectories.npz"),
        os.path.join(data_root, f"{public_pde}_controlled_trajectories.npz"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot find the controlled {pde} dataset under {data_root}. "
        "Use the public datasets/discovery/<pde>_controlled_trajectories.npz layout."
    )


# -----------------------------------------------------------------------------
# Evaluators
# -----------------------------------------------------------------------------
class PairEvaluator:
    def __init__(self, pde: str, model: str, cfg: Config):
        self.pde = pde
        self.model_name = model
        self.cfg = cfg
        script = {
            "burgers": cfg.burgers_script,
            "grayscott": cfg.grayscott_script,
            "kolmogorov": cfg.kolmogorov_script,
        }[pde]
        self.mod = load_module_from_path(f"surrogate_pair_eval_{pde}_{model}_{os.getpid()}", script)
        self.ckpt_path = resolve_checkpoint(cfg.ckpt_root, pde, model)
        self.dataset_path = resolve_dataset(cfg.data_root, pde)
        self.ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.model, self.model_type, self.data_meta = self._build_model()
        self.ckpt_cfg = self.ckpt["cfg"]
        self.exp_cfg = self._make_exp_cfg()
        self._prepare_env()
        # Initial conditions are loaded later by refresh_ics(n_conditions, ic_mode).
        # Avoid loading a default large pool here, which is redundant and can
        # trigger PDE-specific loader issues when only archived targets
        # are being reused.
        self.x0s, self.tags, self.z0s = [], [], []

    def _build_model(self):
        fn = self.mod.build_model_from_ckpt
        sig = inspect.signature(fn)
        kwargs = {}
        if "train_script" in sig.parameters:
            kwargs["train_script"] = self.cfg.train_script
        # coords_2d may not be available before prepare_env; frozen loaders usually accept None.
        if "coords_2d" in sig.parameters:
            kwargs["coords_2d"] = None
        if "expected_pde" in sig.parameters:
            kwargs["expected_pde"] = self.pde
        try:
            return fn(self.ckpt, **kwargs)
        except TypeError:
            return fn(self.ckpt)

    def _make_exp_cfg(self):
        C = self.mod.CFG
        e = C()
        e.ckpt_path = self.ckpt_path
        e.dataset_path = self.dataset_path
        e.H = self.cfg.H
        e.n_eval_ic = 1000  # overwritten later using the decision-archive length
        e.ic_mode = self.cfg.ic_mode
        # preserve defaults from script for dt_true, true-grid, etc.
        return e

    def _prepare_env(self):
        dm = self.data_meta
        if self.pde == "burgers":
            self.nx_nom = int(dm["nx"])
            self.L = float(dm["L"])
            self.nu = float(dm["nu"])
            self.dt_nom = float(dm["dt"])
            self.exp_cfg.umin = float(dm["u_min"])
            self.exp_cfg.umax = float(dm["u_max"])
            self.umin, self.umax = self.exp_cfg.umin, self.exp_cfg.umax
            self.x_nom = self.mod.make_grid(self.nx_nom, self.L)
            self.x_true = self.mod.make_grid(int(getattr(self.exp_cfg, "nx_true", 256)), self.L)
            _, _, self.dx_nom = self.mod.build_periodic_D1_D2(self.nx_nom, self.L)
            self.dx_true = self.L / len(self.x_true)
            self.k_nom = self.mod.control_profile(self.x_nom, float(dm["ctrl_center"]), float(dm["ctrl_sigma"]), self.L)
            self.k_true = self.mod.control_profile(self.x_true, float(dm["ctrl_center"]), float(dm["ctrl_sigma"]), self.L)
            self.x_norm = 2.0 * (self.x_nom / self.L) - 1.0
        else:
            self.nx = int(dm["nx"]); self.ny = int(dm["ny"])
            self.Lx = float(dm["Lx"]); self.Ly = float(dm["Ly"])
            self.dt_nom = float(dm["dt"])
            self.exp_cfg.umin = float(dm["u_min"]); self.exp_cfg.umax = float(dm["u_max"])
            self.umin, self.umax = self.exp_cfg.umin, self.exp_cfg.umax
            self.x, self.y, self.X, self.Y, self.dx, self.dy = self.mod.make_periodic_2d_grid(self.nx, self.ny, self.Lx, self.Ly)
            self.actuator = self.mod.gaussian_profile_2d(
                self.X, self.Y,
                float(dm["ctrl_center_x"]), float(dm["ctrl_center_y"]), float(dm["ctrl_sigma"]),
                self.Lx, self.Ly,
            )
            x_norm = 2.0 * (self.x / self.Lx) - 1.0
            y_norm = 2.0 * (self.y / self.Ly) - 1.0
            Xn, Yn = np.meshgrid(x_norm, y_norm)
            self.coords_2d = np.stack([Xn, Yn], axis=-1).astype(np.float32)
            # Rebuild model if loader wanted coords_2d
            if self.pde == "kolmogorov":
                try:
                    fn = self.mod.build_model_from_ckpt
                    sig = inspect.signature(fn)
                    if "coords_2d" in sig.parameters:
                        kwargs = {"train_script": self.cfg.train_script, "coords_2d": self.coords_2d}
                        if "expected_pde" in sig.parameters:
                            kwargs["expected_pde"] = self.pde
                        self.model, self.model_type, self.data_meta = fn(self.ckpt, **kwargs)
                except Exception:
                    pass
            if self.pde == "grayscott":
                self.Du = float(dm["Du"]); self.Dv = float(dm["Dv"]); self.F0 = float(dm["F"]); self.k0 = float(dm["k"])
            elif self.pde == "kolmogorov":
                self.nu = float(dm["nu"]); self.forcing_amp = float(dm["forcing_amp"]); self.forcing_k = int(dm["forcing_k"])
                # The Kolmogorov and Gray--Scott loaders use different signatures:
                # load_initial_conditions_from_dataset(dataset, ckpt_cfg, n, X, Y, dx, dy, base_field, cfg).
                try:
                    if hasattr(self.mod, "base_kolmogorov_field"):
                        try:
                            self.base_field = self.mod.base_kolmogorov_field(self.X, self.Y, self.forcing_k)
                        except TypeError:
                            self.base_field = self.mod.base_kolmogorov_field(self.X, self.Y, self.forcing_amp, self.forcing_k)
                    else:
                        self.base_field = np.cos(float(self.forcing_k) * self.Y).astype(np.float32)
                except Exception:
                    self.base_field = np.cos(float(self.forcing_k) * self.Y).astype(np.float32)
                self.base_field = np.asarray(self.base_field, dtype=np.float32)

    def _load_ics(self):
        # n_eval_ic is set to the decision-archive length in refresh_ics.
        n = int(getattr(self.exp_cfg, "n_eval_ic", 1000))
        fn = self.mod.load_initial_conditions_from_dataset
        if self.pde == "burgers":
            return fn(self.dataset_path, self.ckpt_cfg, n, self.x_nom, self.dx_nom, self.exp_cfg)
        if self.pde == "grayscott":
            return fn(self.dataset_path, self.ckpt_cfg, n, self.dx, self.dy, self.exp_cfg)
        if self.pde == "kolmogorov":
            if not hasattr(self, "base_field") or self.base_field is None:
                self.base_field = np.cos(float(self.forcing_k) * self.Y).astype(np.float32)
            try:
                return fn(self.dataset_path, self.ckpt_cfg, n, self.X, self.Y, self.dx, self.dy, self.base_field, self.exp_cfg)
            except TypeError as e1:
                try:
                    return fn(self.dataset_path, self.ckpt_cfg, n, self.x, self.y, self.dx, self.dy, self.base_field, self.exp_cfg)
                except TypeError:
                    try:
                        return fn(self.dataset_path, self.ckpt_cfg, n, self.dx, self.dy, self.exp_cfg)
                    except TypeError as e3:
                        raise TypeError(
                            "Failed to call Kolmogorov load_initial_conditions_from_dataset(). "
                            "Tried signatures (dataset, ckpt_cfg, n, X, Y, dx, dy, base_field, cfg), "
                            "(dataset, ckpt_cfg, n, x, y, dx, dy, base_field, cfg), and "
                            "(dataset, ckpt_cfg, n, dx, dy, cfg). Original errors: "
                            f"{e1}; {e3}"
                        )
        raise ValueError(f"Unknown PDE: {self.pde}")

    def refresh_ics(self, n_eval_ic: int, ic_mode: str):
        self.exp_cfg.n_eval_ic = int(n_eval_ic)
        self.exp_cfg.ic_mode = str(ic_mode)
        self.x0s, self.tags, self.z0s = self._load_ics()

    def evaluate(self, x0: np.ndarray, U: np.ndarray, z_limit: float) -> Dict[str, float]:
        U = np.asarray(U, dtype=np.float64).reshape(-1)
        if self.pde == "burgers":
            states_sur = self.mod.surrogate_rollout(self.model, self.model_type, x0, U, self.x_norm)
            z_sur = self.mod.risk_Z_inf(states_sur, self.dx_nom)
            J_sur = self.mod.terminal_response(states_sur[-1], self.k_nom)
            u0_true = self.mod.periodic_interp_1d(x0, self.x_nom, self.x_true)
            states_hf = self.mod.rollout_true(
                u0_true=u0_true, a_seq=U, x_true=self.x_true, k_true=self.k_true,
                nu=self.nu, dt_nom=self.dt_nom, dt_true=float(self.exp_cfg.dt_true)
            )
            z_hf = self.mod.risk_Z_inf(states_hf, self.dx_true)
            J_hf = self.mod.terminal_response(states_hf[-1], self.k_true)
        elif self.pde == "grayscott":
            states_sur = self.mod.surrogate_rollout(self.model, self.model_type, x0, U, self.coords_2d)
            z_sur = self.mod.risk_Z_front(states_sur, self.dx, self.dy)
            J_sur = self.mod.nominal_performance(states_sur, self.actuator)
            states_hf = self.mod.rollout_true(
                s0_true=x0, a_seq=U, actuator=self.actuator, dx=self.dx, dy=self.dy,
                Du=self.Du, Dv=self.Dv, F0=self.F0, k0=self.k0,
                dt_nom=self.dt_nom, dt_true=float(self.exp_cfg.dt_true)
            )
            z_hf = self.mod.risk_Z_front(states_hf, self.dx, self.dy)
            J_hf = self.mod.nominal_performance(states_hf, self.actuator)
        else:
            states_sur = self.mod.surrogate_rollout(self.model, self.model_type, x0, U, self.coords_2d)
            z_sur = self.mod.risk_Z_inf(states_sur, self.dx, self.dy)
            J_sur = self.mod.nominal_performance(states_sur, self.actuator)
            states_hf = self.mod.rollout_true(
                s0_true=x0, a_seq=U, actuator=self.actuator, x=self.x, y=self.y,
                nu=self.nu, dt_nom=self.dt_nom, dt_true=float(self.exp_cfg.dt_true),
                forcing_amp=self.forcing_amp, forcing_k=self.forcing_k,
            )
            z_hf = self.mod.risk_Z_inf(states_hf, self.dx, self.dy)
            J_hf = self.mod.nominal_performance(states_hf, self.actuator)
        return {
            "z_sur": float(z_sur), "z_hf": float(z_hf), "J_sur": float(J_sur), "J_hf": float(J_hf),
            "m_s": float(z_limit - z_sur), "m_hf": float(z_limit - z_hf),
            "gap": float(z_hf - z_sur),
        }


# -----------------------------------------------------------------------------
# Dataset and candidates
# -----------------------------------------------------------------------------
@dataclass
class SampleRecord:
    pde: str
    model: str
    ic_idx: int
    tag: str
    x0: np.ndarray
    U_star: np.ndarray
    z_limit: float
    z_sur: float
    z_hf: float
    J_sur: float
    J_hf: float
    m_s: float
    m_hf: float
    gap: float
    regime: str
    mode: str


def build_features(rec: SampleRecord, Hc: int) -> np.ndarray:
    Uc = interp_1d(rec.U_star, Hc)
    pde_idx = PDE_LIST.index(rec.pde)
    model_idx = MODEL_LIST.index(rec.model)
    reg_idx = REGIME_NAMES.index(rec.regime)
    mode_idx = MODE_NAMES.index(rec.mode)
    scalars = np.asarray([
        rec.m_s, rec.m_hf, rec.gap, max(rec.gap, 0.0), rec.z_sur, rec.z_hf, rec.z_limit,
        rec.J_sur, rec.J_hf, np.mean(Uc), np.std(Uc), np.min(Uc), np.max(Uc), np.mean(Uc**2),
        total_variation(Uc), high_freq_energy_ratio(Uc),
        float(np.mean(np.abs(Uc) > 0.95 * (np.max(np.abs(Uc)) + 1e-12))),
    ], dtype=np.float64)
    return np.concatenate([one_hot(pde_idx, len(PDE_LIST)), one_hot(model_idx, len(MODEL_LIST)), one_hot(reg_idx, 4), one_hot(mode_idx, 4), scalars, Uc], axis=0)


def encode_U(U: np.ndarray, Hc: int, umin: float, umax: float) -> np.ndarray:
    Uc = interp_1d(U, Hc)
    center = 0.5 * (umin + umax)
    scale = max(1e-8, 0.5 * (umax - umin))
    return np.clip((Uc - center) / scale, -1.5, 1.5)


def decode_U(Ucan: np.ndarray, H: int, umin: float, umax: float) -> np.ndarray:
    center = 0.5 * (umin + umax)
    scale = max(1e-8, 0.5 * (umax - umin))
    U = center + scale * np.asarray(Ucan, dtype=np.float64).reshape(-1)
    U = interp_1d(U, H)
    return np.clip(U, umin, umax)


def project_feasible(U: np.ndarray, umin: float, umax: float, smooth_k: int = 1, blend: float = 0.0) -> np.ndarray:
    U = np.clip(np.asarray(U, dtype=np.float64).reshape(-1), umin, umax)
    if smooth_k and smooth_k > 1 and blend > 0:
        Us = moving_average(U, smooth_k)
        U = (1.0 - blend) * U + blend * Us
    return np.clip(U, umin, umax)


def generate_candidates(U: np.ndarray, umin: float, umax: float, cfg: Config, rng: np.random.Generator) -> List[np.ndarray]:
    """Construct the shared candidate bank.

    This bank is deliberately PDE-agnostic. It creates candidates by combining
    low-pass smoothing, amplitude interpolation, bias shifts, multi-scale ramps,
    and random smooth perturbations around u*. The goal is not to tune to a
    particular PDE, but to give the teacher selector a richer set of possible
    safe/performance-preserving corrections. If the teacher upper bound is weak,
    the learned projector cannot succeed.
    """
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    H = len(U)
    center = 0.5 * (umin + umax)
    span = max(1e-8, umax - umin)
    alpha_grid = parse_float_grid(getattr(cfg, "candidate_alpha_grid", ""), [0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0])
    smooth_grid = parse_int_grid(getattr(cfg, "candidate_smooth_grid", ""), [1, 3, 5, 9, 15])
    cand = []

    def add(x):
        x = np.clip(np.asarray(x, dtype=np.float64).reshape(-1), umin, umax)
        if len(x) != H:
            x = interp_1d(x, H)
        cand.append(np.clip(x, umin, umax))

    # Identity is always present, making no-harm possible.
    add(U)

    # High-priority diverse anchors must appear before the candidate-count cap.
    # Earlier versions filled small candidate banks with near-smooth variants,
    # which made safety repair weak when candidate_count was small. These are
    # PDE-agnostic control-shape anchors, not PDE-specific fallbacks.
    add(np.full_like(U, center))
    add(np.full_like(U, umin))
    add(np.full_like(U, umax))
    add(center + 0.25 * (U - center))
    add(center + 0.50 * (U - center))
    add(center + 0.75 * (U - center))
    add(center - 0.25 * (U - center))
    add(center - 0.50 * (U - center))
    add(center - 0.75 * (U - center))
    add(center - 1.00 * (U - center))
    add(np.linspace(U[0], center, H))
    add(np.linspace(center, U[-1], H))
    add(np.linspace(U[0], U[-1], H))
    add(np.linspace(U[0], umin, H))
    add(np.linspace(U[0], umax, H))
    add(np.linspace(umin, U[-1], H))
    add(np.linspace(umax, U[-1], H))
    for sh in [max(1, H // 8), max(1, H // 4), max(1, H // 2)]:
        add(np.roll(U, sh))
        add(np.roll(U, -sh))

    # Low-pass variants and interpolation to them.
    smooth_refs = []
    for k in smooth_grid:
        Us = moving_average(U, k)
        smooth_refs.append(Us)
        add(Us)
        for a in alpha_grid:
            add((1.0 - a) * U + a * Us)

    # Interpolation to center / low-energy references.
    refs = [np.full_like(U, center), 0.0 * U + center]
    # Monotone ramps are universal control-shape priors, not PDE-specific.
    refs += [np.linspace(U[0], center, H), np.linspace(center, U[-1], H)]
    refs += [np.linspace(U[0], U[-1], H)]
    for ref in refs:
        for a in alpha_grid:
            add((1.0 - a) * U + a * ref)

    # Amplitude sweeps around the control center.
    for scale in [-1.0, -0.75, -0.5, -0.25, 0.15, 0.25, 0.4, 0.6, 0.8, 0.9, 1.05, 1.15]:
        add(center + scale * (U - center))
    for bias in [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]:
        add(U + bias * span)

    # Piecewise-constant block averages, useful for suppressing unsafe spikes.
    for blocks in [2, 4, 8, 16]:
        if H >= blocks:
            V = U.copy()
            edges = np.linspace(0, H, blocks + 1).astype(int)
            for i in range(blocks):
                lo, hi = edges[i], edges[i + 1]
                if hi > lo:
                    V[lo:hi] = np.mean(U[lo:hi])
            for a in [0.25, 0.5, 0.75, 1.0]:
                add((1 - a) * U + a * V)

    # Random smooth local perturbations.
    for _ in range(max(0, int(cfg.random_candidates))):
        noise = rng.normal(0.0, cfg.candidate_noise_scale * span, size=H)
        noise = moving_average(noise, max(3, int(cfg.candidate_smooth_k)))
        add(U + noise)
        add(moving_average(U + noise, max(3, int(cfg.candidate_smooth_k))))

    # Unique and capped by candidate_count.
    out = []
    seen = set()
    for x in cand:
        key = tuple(np.round(x, 5).tolist())
        if key not in seen:
            out.append(x)
            seen.add(key)
        if len(out) >= max(2, int(cfg.candidate_count)):
            break
    return out

def capped_margin_reward(m: float, cap: float) -> float:
    """Reward margin only until a universal cap; avoids over-conservative safety inflation."""
    return float(min(max(m, -cap), cap))


def teacher_score(mode: str, before: Dict[str, float], after: Dict[str, float], U: np.ndarray, Ustar: np.ndarray, cfg: Config) -> float:
    """Teacher objective with decision-regime-specific performance semantics.

    The score is not PDE/model-specific. It separates evaluation objectives by
    semantic regime:
      - safety_repair and robust_risk_reduction: safety/risk is primary; report
        performance loss separately instead of averaging it with all samples.
      - performance_recovery: high-fidelity safety plus relative performance gain.
      - identity_minimal: do-no-harm and minimal displacement.
    """
    dist = float(np.sqrt(np.mean((U - Ustar) ** 2)))
    m0 = float(before["m_hf"])
    J0 = float(before["J_hf"])
    m = float(after["m_hf"])
    J = float(after["J_hf"])
    eps = float(cfg.safety_margin_target)
    cap = float(max(cfg.margin_cap, eps))
    dJ_rel = rel_gain(J, J0)
    perf_loss_rel = max(-dJ_rel, 0.0)
    dm = m - m0
    violation = max(eps - m, 0.0)
    hard_unsafe = max(-m, 0.0)

    if mode == "identity_minimal":
        # If already semantically consistent, almost identity should win.
        no_harm = - cfg.no_harm_weight * (max(m0 - m, 0.0) ** 2 + max(perf_loss_rel - cfg.relative_j_tolerance, 0.0) ** 2)
        return no_harm + 0.05 * max(dJ_rel, 0.0) + 0.05 * capped_margin_reward(m, cap) - cfg.teacher_identity_distance_weight * dist

    if mode == "safety_repair":
        # First fix HF safety. Once repaired, performance loss and locality dominate.
        repaired = 1.0 if m >= eps else 0.0
        margin_direction = 2.0 * max(dm, 0.0) - 8.0 * max(-dm, 0.0)
        repair_term = 6.0 * repaired - 12.0 * violation ** 2 + margin_direction
        perf_budget_penalty = -3.0 * max(perf_loss_rel - cfg.performance_loss_budget_safety, 0.0) ** 2
        return repair_term + 0.30 * capped_margin_reward(m, cap) + 0.05 * max(dJ_rel, 0.0) + perf_budget_penalty - cfg.teacher_distance_weight * dist

    if mode == "performance_recovery":
        # Conservative rejection: success is HF-safe relative performance recovery.
        safe_term = 2.0 if m >= 0.0 else -8.0 * hard_unsafe ** 2
        gain_term = 4.0 * max(dJ_rel, 0.0) - 1.0 * max(-dJ_rel, 0.0)
        return safe_term + gain_term + 0.10 * capped_margin_reward(m, cap) - cfg.teacher_distance_weight * dist

    # Unsafe agreement: risk reduction is primary; performance loss is reported separately.
    repaired_bonus = 2.0 if m >= 0.0 else 0.0
    perf_budget_penalty = -1.5 * max(perf_loss_rel - cfg.performance_loss_budget_unsafe, 0.0) ** 2
    margin_direction = 6.0 * max(dm, 0.0) - 12.0 * max(-dm, 0.0)
    return repaired_bonus + margin_direction - 12.0 * violation ** 2 + 0.02 * max(dJ_rel, 0.0) + perf_budget_penalty - cfg.teacher_distance_weight * dist

def select_teacher(rec: SampleRecord, evaluator: PairEvaluator, cfg: Config, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, Any]]:
    if int(getattr(cfg, "force_identity_for_safe_agreement", 1)) and rec.regime == "safe_agreement":
        return rec.U_star.copy(), {
            "best": {
                "cand_id": -1,
                "score": 0.0,
                "dist": 0.0,
                "m_s": rec.m_s,
                "m_hf": rec.m_hf,
                "J_hf": rec.J_hf,
                "z_sur": rec.z_sur,
                "z_hf": rec.z_hf,
                "gap": rec.gap,
            },
            "candidates": [{
                "cand_id": -1,
                "score": 0.0,
                "dist": 0.0,
                "m_s": rec.m_s,
                "m_hf": rec.m_hf,
                "J_hf": rec.J_hf,
            }],
        }
    cands = generate_candidates(rec.U_star, evaluator.umin, evaluator.umax, cfg, rng)
    before = {"m_s": rec.m_s, "m_hf": rec.m_hf, "J_hf": rec.J_hf}
    rows = []
    best = None
    for k, U in enumerate(cands):
        ev = evaluator.evaluate(rec.x0, U, rec.z_limit)
        sc = teacher_score(rec.mode, before, ev, U, rec.U_star, cfg)
        row = {"cand_id": k, "score": float(sc), "dist": float(np.sqrt(np.mean((U-rec.U_star)**2))), **ev}
        rows.append(row)
        if best is None or sc > best[0]:
            best = (sc, U, row)
    assert best is not None
    return best[1], {"best": best[2], "candidates": rows}


def build_records_for_pair(pde: str, model: str, cfg: Config) -> Tuple[List[SampleRecord], PairEvaluator, Dict[str, Any]]:
    run = load_decision_archive(cfg.root, pde, model, cfg.ic_mode)
    arr = run["npz"]
    n = int(len(arr["U_all"]))
    if cfg.max_samples_per_pair and cfg.max_samples_per_pair > 0:
        n = min(n, int(cfg.max_samples_per_pair))
    evaluator = PairEvaluator(pde, model, cfg)
    evaluator.refresh_ics(n, cfg.ic_mode)
    records = []
    for i in range(n):
        U = np.asarray(arr["U_all"][i], dtype=np.float64).reshape(-1)
        x0 = np.asarray(evaluator.x0s[i], dtype=np.float64)
        z_limit = float(arr["z_limit"][i])
        z_sur = float(arr["z_sur"][i])
        z_hf = float(arr["z_hf"][i])
        J_sur = float(arr["J_sur"][i]) if "J_sur" in arr else float(arr.get("J_pred", [np.nan])[i])
        J_hf = float(arr["J_hf"][i]) if "J_hf" in arr else float(arr.get("J_true", [np.nan])[i])
        m_s = float(arr["margin_sur"][i]) if "margin_sur" in arr else z_limit - z_sur
        m_hf = float(arr["margin_hf"][i]) if "margin_hf" in arr else z_limit - z_hf
        gap = float(arr["gap"][i]) if "gap" in arr else z_hf - z_sur
        decision_archive_vals = {
            "z_sur": z_sur, "z_hf": z_hf, "J_sur": J_sur, "J_hf": J_hf,
            "m_s": m_s, "m_hf": m_hf, "gap": gap,
        }
        if int(getattr(cfg, "reevaluate_baseline", 1)):
            # Teacher candidates are evaluated through PairEvaluator. Use the
            # same evaluator for the baseline so identity has exactly zero
            # semantic gain and cannot be dominated by cache/IC mismatch.
            ev0 = evaluator.evaluate(x0, U, z_limit)
            z_sur, z_hf, J_sur, J_hf = float(ev0["z_sur"]), float(ev0["z_hf"]), float(ev0["J_sur"]), float(ev0["J_hf"])
            m_s, m_hf, gap = float(ev0["m_s"]), float(ev0["m_hf"]), float(ev0["gap"])
        reg = classify_regime(m_s, m_hf)
        mode = REGIME_TO_MODE[reg]
        tag = str(arr["tags"][i]) if "tags" in arr else str(i)
        rec = SampleRecord(pde, model, i, tag, x0, U, z_limit, z_sur, z_hf, J_sur, J_hf, m_s, m_hf, gap, reg, mode)
        setattr(rec, "decision_archive_vals", decision_archive_vals)
        records.append(rec)
    meta = {"decision_archive_dir": run["dir"], "decision_archive_npz": run["npz_path"], "checkpoint": evaluator.ckpt_path, "dataset": evaluator.dataset_path, "n": n}
    return records, evaluator, meta


# -----------------------------------------------------------------------------
# Projection model
# -----------------------------------------------------------------------------
class UniversalMLPProjector(nn.Module):
    """Default universal projector: context + canonical control sequence -> correction."""
    def __init__(self, in_dim: int, Hc: int, hidden: int = 256, depth: int = 4, dropout: float = 0.05):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, depth - 1)):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.delta_head = nn.Linear(d, Hc)
        self.trust_head = nn.Linear(d, 1)

    def forward(self, x):
        h = self.backbone(x)
        delta = self.delta_head(h)
        trust = torch.sigmoid(self.trust_head(h)).squeeze(-1)
        return delta, trust


class ContextAttentionProjector(nn.Module):
    """
    Optional sequence-aware projector.

    The input vector is [context, canonical U*]. Context tokens encode PDE/model/regime/mode/scalars;
    U* is treated as a length-Hc sequence. Attention can help when correction depends on temporal
    structure such as switching, saturation, or local pulses. It is slower and needs more data than MLP.
    """
    def __init__(self, in_dim: int, Hc: int, hidden: int = 256, nhead: int = 4, layers: int = 2, dropout: float = 0.05):
        super().__init__()
        self.Hc = Hc
        self.context_dim = in_dim - Hc
        self.hidden = hidden
        self.context = nn.Sequential(
            nn.Linear(self.context_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
        )
        self.u_embed = nn.Linear(2, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(nhead)),
            dim_feedforward=max(hidden * 2, 128),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(layers)))
        self.delta_head = nn.Linear(hidden, 1)
        self.trust_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        ctx = x[:, :self.context_dim]
        u = x[:, self.context_dim:]
        B, H = u.shape
        pos = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)[None, :].expand(B, H)
        token = torch.stack([u, pos], dim=-1)
        h = self.u_embed(token)
        c = self.context(ctx)[:, None, :]
        h = h + c
        h = self.encoder(h)
        delta = self.delta_head(h).squeeze(-1)
        trust = torch.sigmoid(self.trust_head(torch.mean(h, dim=1))).squeeze(-1)
        return delta, trust


def make_projector(in_dim: int, Hc: int, cfg: Config) -> nn.Module:
    if str(cfg.model_arch).lower() in ["attention", "transformer", "attn"]:
        return ContextAttentionProjector(in_dim, Hc, cfg.hidden_dim, cfg.nhead, cfg.transformer_layers, cfg.dropout)
    return UniversalMLPProjector(in_dim, Hc, cfg.hidden_dim, cfg.depth, cfg.dropout)

def compute_projector_loss(model, Xt, Yt, Wt, idx, cfg: Config):
    pred, trust = model(Xt[idx])
    max_norm = cfg.trust_min + (cfg.trust_max - cfg.trust_min) * trust[:, None]
    pred_clip = torch.tanh(pred / (max_norm + 1e-6)) * max_norm
    mse = torch.mean((pred_clip - Yt[idx]) ** 2, dim=1)
    smooth = torch.mean((pred_clip[:, 1:] - pred_clip[:, :-1]) ** 2, dim=1)
    loss = torch.mean(Wt[idx] * (cfg.lambda_control_mse * mse + cfg.lambda_smooth * smooth))
    return loss


def train_model(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
    cfg: Config,
    train_local_idx: np.ndarray,
    val_local_idx: np.ndarray,
) -> Tuple[nn.Module, Dict[str, Any], Tuple[np.ndarray, np.ndarray]]:
    med, scale = robust_scale_fit(X[train_local_idx])
    Xn = robust_scale_apply(X, med, scale)
    device = torch.device(cfg.device)
    model = make_projector(Xn.shape[1], cfg.canonical_horizon, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y.astype(np.float32), dtype=torch.float32, device=device)
    Wt = torch.tensor(weights.astype(np.float32), dtype=torch.float32, device=device)
    rng = np.random.default_rng(cfg.seed)
    hist = []
    best_state = copy_state_dict(model)
    best_val = float("inf")
    best_epoch = 0
    bad = 0
    train_local_idx = np.asarray(train_local_idx, dtype=int)
    val_local_idx = np.asarray(val_local_idx, dtype=int)
    if len(val_local_idx) == 0:
        val_local_idx = train_local_idx.copy()

    for ep in tqdm(range(1, cfg.epochs + 1), desc=f"[train] universal semantic projector ({cfg.model_arch})"):
        model.train()
        idx_all = train_local_idx.copy()
        rng.shuffle(idx_all)
        losses = []
        for s in range(0, len(idx_all), cfg.batch_size):
            idx = idx_all[s:s + cfg.batch_size]
            opt.zero_grad()
            loss = compute_projector_loss(model, Xt, Yt, Wt, idx, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else np.nan
        model.eval()
        with torch.no_grad():
            val_loss = float(compute_projector_loss(model, Xt, Yt, Wt, val_local_idx, cfg).item())
        hist.append({"epoch": ep, "loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - float(cfg.min_delta):
            best_val = val_loss
            best_epoch = ep
            best_state = copy_state_dict(model)
            bad = 0
        else:
            bad += 1
        if cfg.early_stopping_patience > 0 and bad >= cfg.early_stopping_patience:
            break
    model.load_state_dict(best_state)
    return model, {"history": hist, "best_epoch": int(best_epoch), "best_val_loss": float(best_val), "model_arch": cfg.model_arch}, (med, scale)


def copy_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def predict_project(model: UniversalProjector, feat: np.ndarray, norm: Tuple[np.ndarray, np.ndarray], U_star: np.ndarray,
                    Hc: int, H: int, umin: float, umax: float, cfg: Config) -> np.ndarray:
    med, scale = norm
    x = robust_scale_apply(feat[None, :], med, scale)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        delta, trust = model(torch.tensor(x, dtype=torch.float32, device=device))
        max_norm = cfg.trust_min + (cfg.trust_max - cfg.trust_min) * trust[:, None]
        delta = torch.tanh(delta / (max_norm + 1e-6)) * max_norm
        delta = delta[0].cpu().numpy().astype(np.float64)
    Ucan = encode_U(U_star, Hc, umin, umax) + delta
    Unew = decode_U(Ucan, H, umin, umax)
    Unew = project_feasible(Unew, umin, umax, smooth_k=cfg.candidate_smooth_k, blend=0.25)
    return Unew


# -----------------------------------------------------------------------------
# Metrics and plotting
# -----------------------------------------------------------------------------
def assign_generalization_splits(rows_meta: List[Dict[str, Any]], pairs: List[Tuple[str, str]], cfg: Config) -> None:
    """Assign train/val/test splits for generalization evaluation.

    split_mode=ic: stratified IC-level split inside every PDE/model pair.
    split_mode=pair: use --train_pairs/--test_pairs if provided; otherwise random pair holdout.
    split_mode=leave_pde: test all samples from --holdout_pde.
    split_mode=leave_model: test all samples from --holdout_model.
    """
    rng = np.random.default_rng(cfg.seed)
    pair_keys = [pair_key(*p) for p in pairs]
    train_pairs = set(pair_key(*p) for p in parse_pairs(cfg.train_pairs)) if cfg.train_pairs else set(pair_keys)
    test_pairs = set(pair_key(*p) for p in parse_pairs(cfg.test_pairs)) if cfg.test_pairs else set()

    for r in rows_meta:
        r["split"] = "train"

    mode = str(cfg.split_mode).lower()
    if cfg.train_pairs or cfg.test_pairs:
        for r in rows_meta:
            k = r["pair"]
            if k in test_pairs and k not in train_pairs:
                r["split"] = "test"
            elif k in test_pairs and k in train_pairs:
                # same pair in both means IC-level split will handle it below
                r["split"] = "train"
            elif k not in train_pairs:
                r["split"] = "ignore"
        # If pairs overlap, still create IC split within those pairs.
        mode = "ic"

    if mode == "leave_pde" and cfg.holdout_pde:
        hp = cfg.holdout_pde.lower()
        for r in rows_meta:
            r["split"] = "test" if r["pde"] == hp else "train"
    elif mode == "leave_model" and cfg.holdout_model:
        hm = cfg.holdout_model.lower()
        for r in rows_meta:
            r["split"] = "test" if r["model"] == hm else "train"
    elif mode == "pair" and not (cfg.train_pairs or cfg.test_pairs):
        shuffled = pair_keys.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * cfg.test_fraction)))
        test_set = set(shuffled[:n_test])
        for r in rows_meta:
            r["split"] = "test" if r["pair"] in test_set else "train"
    elif mode == "ic":
        # stratified within pair and regime for robust generalization evaluation.
        groups: Dict[Tuple[str, str], List[int]] = {}
        for i, r in enumerate(rows_meta):
            if r.get("split") == "ignore":
                continue
            groups.setdefault((r["pair"], r["regime_before"]), []).append(i)
        for _, idxs in groups.items():
            idxs = np.asarray(idxs, dtype=int)
            rng.shuffle(idxs)
            n = len(idxs)
            n_test = int(round(n * cfg.test_fraction))
            n_val = int(round(n * cfg.val_fraction))
            if n >= 5:
                n_test = max(1, n_test)
                n_val = max(1, n_val)
            test_ids = set(idxs[:n_test].tolist())
            val_ids = set(idxs[n_test:n_test+n_val].tolist())
            for i in idxs:
                if int(i) in test_ids:
                    rows_meta[int(i)]["split"] = "test"
                elif int(i) in val_ids:
                    rows_meta[int(i)]["split"] = "val"
                else:
                    rows_meta[int(i)]["split"] = "train"
    # Ensure at least one validation sample unless no training.
    if not any(r.get("split") == "val" for r in rows_meta if r.get("split") != "ignore"):
        train_ids = [i for i, r in enumerate(rows_meta) if r.get("split") == "train"]
        if len(train_ids) > 4:
            rng.shuffle(train_ids)
            for i in train_ids[:max(1, int(0.1 * len(train_ids)))]:
                rows_meta[i]["split"] = "val"


def split_indices(rows_meta: List[Dict[str, Any]], usable_pairs: Optional[set] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train, val, test = [], [], []
    for i, r in enumerate(rows_meta):
        if usable_pairs is not None and r["pair"] not in usable_pairs:
            continue
        sp = r.get("split", "train")
        if sp == "train":
            train.append(i)
        elif sp == "val":
            val.append(i)
        elif sp == "test":
            test.append(i)
    return np.asarray(train, dtype=int), np.asarray(val, dtype=int), np.asarray(test, dtype=int)


def summarize_pair(rows: List[Dict[str, Any]], cfg: Optional[Config] = None) -> Dict[str, float]:
    """Regime-conditioned projection metrics.

    This evaluator avoids using a single mean performance change as the main
    metric. For unsafe/false-safe controls, safety repair and the incurred
    performance loss are reported separately. For conservative rejection, HF-safe
    performance recovery is reported. For safe agreement, do-no-harm preservation
    is reported.
    """
    rel_tol = float(getattr(cfg, "relative_j_tolerance", 1e-3)) if cfg is not None else 1e-3
    perf_budget_safety = float(getattr(cfg, "performance_loss_budget_safety", 0.05)) if cfg is not None else 0.05
    perf_budget_unsafe = float(getattr(cfg, "performance_loss_budget_unsafe", 0.10)) if cfg is not None else 0.10

    def finite(k, subset=None):
        rr = rows if subset is None else subset
        xs = []
        for r in rr:
            if k in r:
                try:
                    v = float(r[k])
                    if np.isfinite(v): xs.append(v)
                except Exception: pass
        return xs
    def mean(k, subset=None):
        xs = finite(k, subset)
        return float(np.mean(xs)) if xs else np.nan
    def median(k, subset=None):
        xs = finite(k, subset)
        return float(np.median(xs)) if xs else np.nan
    def q90(k, subset=None):
        xs = finite(k, subset)
        return float(np.percentile(xs, 90)) if xs else np.nan
    def rate(subset, cond):
        if not subset: return np.nan
        return float(np.mean([bool(cond(r)) for r in subset]))

    before_reg = {name: 0 for name in REGIME_NAMES}
    after_reg = {name: 0 for name in REGIME_NAMES}
    for r in rows:
        before_reg[r["regime_before"]] += 1
        after_reg[r["regime_after"]] += 1
    n = max(1, len(rows))
    by_reg = {reg: [r for r in rows if r["regime_before"] == reg] for reg in REGIME_NAMES}

    fs = by_reg["false_safe_optimism"]
    cons = by_reg["conservative_rejection"]
    unsafe = by_reg["unsafe_agreement"]
    safe = by_reg["safe_agreement"]

    out = {
        "n": len(rows),
        "safe_before": before_reg["safe_agreement"] / n,
        "safe_after": after_reg["safe_agreement"] / n,
        "false_safe_before": before_reg["false_safe_optimism"] / n,
        "false_safe_after": after_reg["false_safe_optimism"] / n,
        "conservative_before": before_reg["conservative_rejection"] / n,
        "conservative_after": after_reg["conservative_rejection"] / n,
        "unsafe_before": before_reg["unsafe_agreement"] / n,
        "unsafe_after": after_reg["unsafe_agreement"] / n,
        # Keep global safety/margin summaries, but do not use global mean J as main evidence.
        "mean_gain_m_hf_all": mean("gain_m_hf"),
        "projected_control_distance_all": mean("dist_projected"),
        # Safe agreement: do-no-harm.
        "safe_n": len(safe),
        "safe_preserve_rate": rate(safe, lambda r: float(r["m_hf_after"]) >= 0.0 and float(r.get("rel_gain_J_hf", 0.0)) >= -rel_tol),
        "safe_mean_perf_loss_rel": mean("perf_loss_rel", safe),
        "safe_q90_perf_loss_rel": q90("perf_loss_rel", safe),
        "safe_mean_control_distance": mean("dist_projected", safe),
        # False-safe: repair is primary; performance loss is reported, not averaged with all points.
        "false_safe_n": len(fs),
        "false_safe_repair_rate": rate(fs, lambda r: float(r["m_hf_after"]) >= 0.0),
        "false_safe_mean_gain_m_hf": mean("gain_m_hf", fs),
        "false_safe_mean_perf_loss_rel": mean("perf_loss_rel", fs),
        "false_safe_perf_loss_within_budget_rate": rate(fs, lambda r: float(r.get("perf_loss_rel", 0.0)) <= perf_budget_safety),
        # Conservative rejection: performance recovery under HF-safety.
        "conservative_n": len(cons),
        "conservative_hf_safe_perf_recovery_rate": rate(cons, lambda r: float(r["m_hf_after"]) >= 0.0 and float(r.get("rel_gain_J_hf", 0.0)) > rel_tol),
        "conservative_hf_safe_rate_after": rate(cons, lambda r: float(r["m_hf_after"]) >= 0.0),
        "conservative_mean_rel_gain_J_hf": mean("rel_gain_J_hf", cons),
        "conservative_median_rel_gain_J_hf": median("rel_gain_J_hf", cons),
        # Unsafe: risk reduction is primary; performance loss separately.
        "unsafe_n": len(unsafe),
        "unsafe_risk_reduction_rate": rate(unsafe, lambda r: float(r["gain_m_hf"]) > 0.0),
        "unsafe_repair_to_hf_safe_rate": rate(unsafe, lambda r: float(r["m_hf_after"]) >= 0.0),
        "unsafe_mean_gain_m_hf": mean("gain_m_hf", unsafe),
        "unsafe_mean_perf_loss_rel": mean("perf_loss_rel", unsafe),
        "unsafe_perf_loss_within_budget_rate": rate(unsafe, lambda r: float(r.get("perf_loss_rel", 0.0)) <= perf_budget_unsafe),
        # Teacher upper bound diagnostics.
        "teacher_mean_gain_m_hf_all": mean("teacher_gain_m_hf"),
        "teacher_mean_rel_gain_J_all": mean("teacher_rel_gain_J_hf"),
        "teacher_false_safe_mean_gain_m_hf": mean("teacher_gain_m_hf", fs),
        "teacher_conservative_mean_rel_gain_J": mean("teacher_rel_gain_J_hf", cons),
        "teacher_unsafe_mean_gain_m_hf": mean("teacher_gain_m_hf", unsafe),
    }
    return out

def nature_style(font_size: int = 8):
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def savefig(fig, path_base: str):
    fig.savefig(path_base + ".png", bbox_inches="tight")
    fig.savefig(path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_margin_arrows(rows: List[Dict[str, Any]], outdir: str, cfg: Config):
    nature_style()
    for pde in sorted(set(r["pde"] for r in rows)):
        rr = [r for r in rows if r["pde"] == pde]
        models = sorted(set(r["model"] for r in rr), key=lambda x: MODEL_LIST.index(x))
        fig, axes = plt.subplots(1, len(models), figsize=(4.0*len(models), 3.4), dpi=cfg.dpi, squeeze=False)
        for ax, model in zip(axes[0], models):
            mm = [r for r in rr if r["model"] == model]
            for r in mm:
                ax.arrow(r["m_s_before"], r["m_hf_before"], r["m_s_after"]-r["m_s_before"], r["m_hf_after"]-r["m_hf_before"],
                         length_includes_head=True, head_width=0.01, alpha=0.35, lw=0.7)
            ax.axhline(0, ls="--", lw=1.0); ax.axvline(0, ls="--", lw=1.0)
            ax.set_title(f"{pde}/{model}")
            ax.set_xlabel(r"$m_s$")
            ax.set_ylabel(r"$m_{HF}$")
            ax.grid(True, alpha=0.2)
        fig.suptitle("Semantic projection before/after in margin plane")
        savefig(fig, os.path.join(outdir, f"fig_projection_margin_arrows_{pde}"))


def plot_pair_heatmaps(pair_summary: Dict[str, Dict[str, float]], outdir: str, cfg: Config):
    nature_style()
    pdes = PDE_LIST
    models = MODEL_LIST
    metrics = [
        ("false_safe_repair_rate", "False-safe repair"),
        ("false_safe_mean_perf_loss_rel", "FS perf. loss"),
        ("conservative_hf_safe_perf_recovery_rate", "Conservative recovery"),
        ("conservative_mean_rel_gain_J_hf", "Conservative rel. perf. gain"),
        ("unsafe_risk_reduction_rate", "Unsafe risk reduction"),
        ("unsafe_mean_perf_loss_rel", "Unsafe perf. loss"),
        ("safe_preserve_rate", "Safe no-harm preserve"),
        ("teacher_mean_gain_m_hf_all", r"Teacher gain $m_{HF}$"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.5), dpi=cfg.dpi)
    for ax, (key, title) in zip(axes.ravel(), metrics):
        M = np.full((len(pdes), len(models)), np.nan)
        for i, pde in enumerate(pdes):
            for j, model in enumerate(models):
                d = pair_summary.get(pair_key(pde, model))
                if d:
                    M[i, j] = d.get(key, np.nan)
        im = ax.imshow(M, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(models))); ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pdes))); ax.set_yticklabels(pdes)
        for i in range(len(pdes)):
            for j in range(len(models)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, os.path.join(outdir, "fig_projection_pair_heatmaps"))


def plot_regime_action(rows: List[Dict[str, Any]], outdir: str, cfg: Config):
    nature_style()
    data = []
    labels = []
    for reg in REGIME_NAMES:
        rr = [r for r in rows if r["regime_before"] == reg]
        if rr:
            data.append([float(r["gain_m_hf"]) for r in rr])
            labels.append(reg.replace("_", "\n"))
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=cfg.dpi)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(0, ls="--", lw=1.0)
    ax.set_ylabel(r"Gain in $m_{HF}$ after projection")
    ax.set_title("Projection effect by decision regime")
    savefig(fig, os.path.join(outdir, "fig_projection_gain_by_regime"))


def plot_control_examples(rows: List[Dict[str, Any]], sample_payloads: Dict[int, Dict[str, Any]], outdir: str, cfg: Config):
    nature_style()
    reps = []
    for reg in ["false_safe_optimism", "conservative_rejection", "unsafe_agreement", "safe_agreement"]:
        rr = [r for r in rows if r["regime_before"] == reg]
        if rr:
            rr = sorted(rr, key=lambda r: abs(float(r["gain_m_hf"])) + abs(float(r["gain_J_hf"])), reverse=True)
            reps.append(rr[0])
    if not reps:
        return
    fig, axes = plt.subplots(len(reps), 1, figsize=(7.2, 2.2*len(reps)), dpi=cfg.dpi, squeeze=False)
    for ax, r in zip(axes[:,0], reps):
        payload = sample_payloads[int(r["global_id"])]
        ax.plot(payload["U_star"], lw=1.5, label=r"$u^*$")
        ax.plot(payload["U_teacher"], lw=1.2, label="teacher")
        ax.plot(payload["U_proj"], lw=1.5, label="projected")
        ax.set_title(f"{r['pde']}/{r['model']} | {r['regime_before']} -> {r['regime_after']}")
        ax.set_ylabel("control")
        ax.grid(True, alpha=0.2)
    axes[-1,0].set_xlabel("time step")
    axes[0,0].legend(frameon=False, ncol=3)
    savefig(fig, os.path.join(outdir, "fig_projection_control_examples"))


def plot_regime_specific_dashboard(pair_summary: Dict[str, Dict[str, float]], outdir: str, cfg: Config):
    """Compact NC-style dashboard that separates safety repair, performance recovery, and no-harm."""
    nature_style()
    pairs_sorted = sorted(pair_summary.keys())
    labels = [k.replace(":", "\n") for k in pairs_sorted]
    metrics = [
        ("safe_preserve_rate", "safe preserve"),
        ("false_safe_repair_rate", "false-safe repair"),
        ("conservative_hf_safe_perf_recovery_rate", "conservative recovery"),
        ("unsafe_risk_reduction_rate", "unsafe risk reduction"),
        ("false_safe_mean_perf_loss_rel", "FS perf. loss"),
        ("unsafe_mean_perf_loss_rel", "unsafe perf. loss"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.2), dpi=cfg.dpi)
    for ax, (key, title) in zip(axes.ravel(), metrics):
        vals = [pair_summary[k].get(key, np.nan) for k in pairs_sorted]
        ax.bar(np.arange(len(vals)), vals)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    savefig(fig, os.path.join(outdir, "fig_projection_regime_specific_dashboard"))


def plot_teacher_upper_bound(pair_summary: Dict[str, Dict[str, float]], outdir: str, cfg: Config):
    nature_style()
    pairs_sorted = sorted(pair_summary.keys())
    labels = [k.replace(":", "\n") for k in pairs_sorted]
    metrics = [
        ("teacher_mean_gain_m_hf_all", r"teacher mean gain $m_{HF}$"),
        ("teacher_false_safe_mean_gain_m_hf", r"teacher FS gain $m_{HF}$"),
        ("teacher_conservative_mean_rel_gain_J", "teacher conservative rel. perf gain"),
        ("teacher_unsafe_mean_gain_m_hf", r"teacher unsafe gain $m_{HF}$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2), dpi=cfg.dpi)
    for ax, (key, title) in zip(axes.ravel(), metrics):
        vals = [pair_summary[k].get(key, np.nan) for k in pairs_sorted]
        ax.bar(np.arange(len(vals)), vals)
        ax.axhline(0, lw=1.0, ls="--")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    savefig(fig, os.path.join(outdir, "fig_projection_teacher_upper_bound"))


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./external_data/decision_archives/discovery")
    ap.add_argument("--ckpt_root", type=str, default="./external_data/checkpoints/discovery")
    ap.add_argument("--data_root", type=str, default="./external_data/datasets/discovery")
    ap.add_argument("--train_script", type=str, default="./scripts/train_discovery_surrogates.py")
    ap.add_argument("--burgers_script", type=str, default="./scripts/generate_burgers_decisions.py")
    ap.add_argument("--grayscott_script", type=str, default="./scripts/generate_gray_scott_decisions.py")
    ap.add_argument("--kolmogorov_script", type=str, default="./scripts/generate_kolmogorov_decisions.py")
    ap.add_argument("--outdir", type=str, default="./semantic_projection_bridge_universal_results")
    ap.add_argument("--ic_mode", type=str, default="challenge_conditions")
    ap.add_argument("--pairs", type=str, default="")
    ap.add_argument("--run_suite", action="store_true")
    ap.add_argument("--pde_list", type=str, default="burgers,grayscott,kolmogorov")
    ap.add_argument("--model_list", type=str, default="deeponet,fno,pino,pinn")
    ap.add_argument("--train_pairs", type=str, default="")
    ap.add_argument("--test_pairs", type=str, default="")
    ap.add_argument("--split_mode", type=str, default="ic", choices=["ic", "pair", "leave_pde", "leave_model"])
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--test_fraction", type=float, default=0.20)
    ap.add_argument("--holdout_pde", type=str, default="")
    ap.add_argument("--holdout_model", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--H", type=int, default=80)
    ap.add_argument("--canonical_horizon", type=int, default=80)
    ap.add_argument("--max_samples_per_pair", type=int, default=-1)
    ap.add_argument("--candidate_count", type=int, default=16)
    ap.add_argument("--random_candidates", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--early_stopping_patience", type=int, default=40)
    ap.add_argument("--min_delta", type=float, default=1e-5)
    ap.add_argument("--model_arch", type=str, default="mlp", choices=["mlp", "attention"])
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--transformer_layers", type=int, default=2)
    ap.add_argument("--trust_min", type=float, default=0.02)
    ap.add_argument("--trust_max", type=float, default=0.85)
    ap.add_argument("--candidate_noise_scale", type=float, default=0.08)
    ap.add_argument("--candidate_smooth_k", type=int, default=7)
    ap.add_argument("--safety_margin_target", type=float, default=0.02)
    ap.add_argument("--margin_cap", type=float, default=0.08)
    ap.add_argument("--j_tolerance", type=float, default=1e-4)
    ap.add_argument("--no_harm_weight", type=float, default=8.0)
    ap.add_argument("--teacher_identity_distance_weight", type=float, default=20.0)
    ap.add_argument("--teacher_distance_weight", type=float, default=2.0)
    ap.add_argument("--relative_j_tolerance", type=float, default=1e-3)
    ap.add_argument("--performance_loss_budget_safety", type=float, default=0.05)
    ap.add_argument("--performance_loss_budget_unsafe", type=float, default=0.10)
    ap.add_argument("--performance_gain_target", type=float, default=1e-3)
    ap.add_argument("--candidate_alpha_grid", type=str, default="0,0.05,0.10,0.20,0.35,0.50,0.70,0.90,1.0")
    ap.add_argument("--candidate_smooth_grid", type=str, default="1,3,5,9,15")
    ap.add_argument("--force_identity_for_safe_agreement", type=int, default=1)
    ap.add_argument("--reevaluate_baseline", type=int, default=1)
    ap.add_argument("--no_train", type=int, default=0)
    ap.add_argument("--rebuild_cache", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=360)
    args = ap.parse_args()
    cfg = Config(**vars(args))
    set_seed(cfg.seed)
    ensure_dir(cfg.outdir)
    cache_path = os.path.join(cfg.outdir, "projection_teacher_cache.npz")

    pairs = build_pairs_from_cfg(cfg)
    explicit_train_pairs = parse_pairs(cfg.train_pairs) if cfg.train_pairs else []
    explicit_test_pairs = parse_pairs(cfg.test_pairs) if cfg.test_pairs else []
    train_set = set(pair_key(*p) for p in explicit_train_pairs) if explicit_train_pairs else set(pair_key(*p) for p in pairs)
    test_set = set(pair_key(*p) for p in explicit_test_pairs) if explicit_test_pairs else set(pair_key(*p) for p in pairs)

    all_samples: List[SampleRecord] = []
    evaluators: Dict[str, PairEvaluator] = {}
    meta = {"pairs": [], "cfg": asdict(cfg)}
    rng = np.random.default_rng(cfg.seed)

    # Build records and teacher targets, with cache.
    if os.path.exists(cache_path) and not cfg.rebuild_cache:
        cache = np.load(cache_path, allow_pickle=True)
        X = cache["X"].astype(np.float64)
        Y = cache["Y"].astype(np.float64)
        weights = cache["weights"].astype(np.float64)
        rows_meta = list(cache["rows_meta"].tolist())
        sample_payloads = {int(k): v for k, v in cache["payloads"].item().items()}
        print(f"[Cache] Loaded teacher cache: {cache_path} | N={len(rows_meta)}")
    else:
        X_list, Y_list, W_list, rows_meta = [], [], [], []
        sample_payloads: Dict[int, Dict[str, Any]] = {}
        gid = 0
        for pde, model in tqdm(pairs, desc="[stage] build teacher targets"):
            recs, evaluator, m = build_records_for_pair(pde, model, cfg)
            evaluators[pair_key(pde, model)] = evaluator
            meta["pairs"].append(m)
            for rec in tqdm(recs, desc=f"[teacher] {pde}/{model}", leave=False):
                U_teacher, info = select_teacher(rec, evaluator, cfg, rng)
                feat = build_features(rec, cfg.canonical_horizon)
                U_star_can = encode_U(rec.U_star, cfg.canonical_horizon, evaluator.umin, evaluator.umax)
                U_teacher_can = encode_U(U_teacher, cfg.canonical_horizon, evaluator.umin, evaluator.umax)
                delta = U_teacher_can - U_star_can
                # regime/mode weights: focus more on corrective regimes
                mode_weight = {"identity_minimal": 0.5, "safety_repair": 2.0, "performance_recovery": 1.5, "robust_risk_reduction": 1.8}[rec.mode]
                row = {
                    "global_id": gid, "pde": rec.pde, "model": rec.model, "pair": pair_key(rec.pde, rec.model),
                    "ic_idx": rec.ic_idx, "tag": rec.tag, "regime_before": rec.regime, "mode": rec.mode,
                    "m_s_before": rec.m_s, "m_hf_before": rec.m_hf, "gap_before": rec.gap,
                    "z_sur_before": rec.z_sur, "z_hf_before": rec.z_hf, "J_sur_before": rec.J_sur, "J_hf_before": rec.J_hf,
                    "archive_m_s_before": getattr(rec, "decision_archive_vals", {}).get("m_s", rec.m_s),
                    "archive_m_hf_before": getattr(rec, "decision_archive_vals", {}).get("m_hf", rec.m_hf),
                    "archive_gap_before": getattr(rec, "decision_archive_vals", {}).get("gap", rec.gap),
                    "archive_J_sur_before": getattr(rec, "decision_archive_vals", {}).get("J_sur", rec.J_sur),
                    "archive_J_hf_before": getattr(rec, "decision_archive_vals", {}).get("J_hf", rec.J_hf),
                    "teacher_m_s": info["best"]["m_s"], "teacher_m_hf": info["best"]["m_hf"],
                    "teacher_J_hf": info["best"]["J_hf"], "teacher_gap": info["best"]["gap"],
                    "teacher_regime": classify_regime(info["best"]["m_s"], info["best"]["m_hf"]),
                    "teacher_dist": info["best"]["dist"], "teacher_score": info["best"]["score"],
                    "teacher_gain_m_hf": float(info["best"]["m_hf"] - rec.m_hf),
                    "teacher_gain_J_hf": float(info["best"]["J_hf"] - rec.J_hf),
                    "teacher_rel_gain_J_hf": rel_gain(info["best"]["J_hf"], rec.J_hf),
                    "teacher_perf_loss_rel": rel_loss(info["best"]["J_hf"], rec.J_hf),
                    "split": "unassigned",
                }
                X_list.append(feat); Y_list.append(delta); W_list.append(mode_weight); rows_meta.append(row)
                sample_payloads[gid] = {"U_star": rec.U_star, "U_teacher": U_teacher, "x0": rec.x0}
                gid += 1
        X = np.asarray(X_list, dtype=np.float64)
        Y = np.asarray(Y_list, dtype=np.float64)
        weights = np.asarray(W_list, dtype=np.float64)
        np.savez_compressed(cache_path, X=X, Y=Y, weights=weights, rows_meta=np.asarray(rows_meta, dtype=object), payloads=sample_payloads)
        save_json(meta, os.path.join(cfg.outdir, "build_meta.json"))
        print(f"[Cache] Saved teacher cache: {cache_path} | N={len(rows_meta)}")

    if cfg.no_train:
        print("[Info] --no_train=1, only teacher cache was built.")
        return

    # Assign train/val/test splits for generalization evaluation.
    assign_generalization_splits(rows_meta, pairs, cfg)
    train_idx, val_idx, test_idx = split_indices(rows_meta, usable_pairs=None)
    if len(test_idx) == 0:
        test_idx = val_idx.copy() if len(val_idx) else train_idx.copy()
    if len(train_idx) == 0:
        raise RuntimeError("No training samples after split assignment. Check --split_mode/--train_pairs/--test_pairs.")
    local_all = np.concatenate([train_idx, val_idx]) if len(val_idx) else train_idx.copy()
    # For normalization and tensors we pass the whole cache, but train_model sees explicit local indices.
    model, train_info, norm = train_model(X, Y, weights, cfg, train_idx, val_idx)
    split_report = {
        "n_total": int(len(rows_meta)), "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "split_mode": cfg.split_mode,
        "train_pairs": sorted(list(set(rows_meta[i]["pair"] for i in train_idx))) if len(train_idx) else [],
        "test_pairs": sorted(list(set(rows_meta[i]["pair"] for i in test_idx))) if len(test_idx) else [],
    }
    save_json(split_report, os.path.join(cfg.outdir, "split_report.json"))

    torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg), "norm_med": norm[0], "norm_scale": norm[1]}, os.path.join(cfg.outdir, "surrogate_pair_evaluator.pt"))
    save_json(train_info, os.path.join(cfg.outdir, "train_info.json"))

    # Need evaluators for all pairs that actually appear in the test split.
    # Do not use a non-existent variable named `test_pairs`; infer the required
    # pair keys directly from rows_meta/test_idx so this works for ic, pair,
    # leave_pde, leave_model, and explicit train/test pair splits.
    eval_pair_keys = sorted(set(str(rows_meta[int(i)]["pair"]) for i in test_idx))
    for k in eval_pair_keys:
        if k not in evaluators:
            pde, model_name = k.split(":", 1)
            _, ev, _ = build_records_for_pair(pde, model_name, cfg)
            evaluators[k] = ev

    rows_out: List[Dict[str, Any]] = []
    for idx in tqdm(test_idx, desc="[eval] projected controls"):
        r = dict(rows_meta[int(idx)])
        pde, model_name = r["pde"], r["model"]
        evaluator = evaluators[pair_key(pde, model_name)]
        # reconstruct record info from rows_meta and payload
        gid = int(r["global_id"])
        U_star = np.asarray(sample_payloads[gid]["U_star"], dtype=np.float64)
        x0 = np.asarray(sample_payloads[gid].get("x0", evaluator.x0s[int(r["ic_idx"])]), dtype=np.float64)
        z_limit = float(r["z_hf_before"] + r["m_hf_before"])
        feat = X[int(idx)]
        identity_path = int(getattr(cfg, "force_identity_for_safe_agreement", 1)) and str(r.get("regime_before", "")) == "safe_agreement"
        if identity_path:
            U_proj = U_star.copy()
            # The online no-harm branch applies exact identity. Re-running the
            # Re-evaluation can differ slightly because reconstructed states are
            # not bit-identical to the archived decision record.
            # For identity, the deployed control sequence is unchanged, so the
            # semantic before metrics are the correct no-harm reference.
            ev = {
                "m_s": float(r["m_s_before"]),
                "m_hf": float(r["m_hf_before"]),
                "gap": float(r["gap_before"]),
                "z_sur": float(r["z_sur_before"]),
                "z_hf": float(r["z_hf_before"]),
                "J_sur": float(r["J_sur_before"]),
                "J_hf": float(r["J_hf_before"]),
            }
        else:
            U_proj = predict_project(model, feat, norm, U_star, cfg.canonical_horizon, len(U_star), evaluator.umin, evaluator.umax, cfg)
            ev = evaluator.evaluate(x0, U_proj, z_limit)
        r.update({
            "m_s_after": ev["m_s"], "m_hf_after": ev["m_hf"], "gap_after": ev["gap"],
            "z_sur_after": ev["z_sur"], "z_hf_after": ev["z_hf"], "J_sur_after": ev["J_sur"], "J_hf_after": ev["J_hf"],
            "regime_after": classify_regime(ev["m_s"], ev["m_hf"]),
            "gain_m_hf": ev["m_hf"] - float(r["m_hf_before"]),
            "gain_J_hf": ev["J_hf"] - float(r["J_hf_before"]),
            "rel_gain_J_hf": rel_gain(ev["J_hf"], float(r["J_hf_before"])),
            "perf_loss_rel": rel_loss(ev["J_hf"], float(r["J_hf_before"])),
            "gain_z_hf": float(r["z_hf_before"]) - ev["z_hf"],
            "dist_projected": float(np.sqrt(np.mean((U_proj - U_star)**2))),
            "identity_path": int(identity_path),
        })
        rows_out.append(r)
        sample_payloads[gid]["U_proj"] = U_proj

    save_csv(rows_out, os.path.join(cfg.outdir, "projection_per_ic.csv"))
    pair_summary = {}
    for k in sorted(set(r["pair"] for r in rows_out)):
        pair_summary[k] = summarize_pair([r for r in rows_out if r["pair"] == k], cfg)
    save_json(pair_summary, os.path.join(cfg.outdir, "projection_pair_summary.json"))

    # figures
    plot_dir = os.path.join(cfg.outdir, "figures")
    ensure_dir(plot_dir)
    plot_margin_arrows(rows_out, plot_dir, cfg)
    plot_pair_heatmaps(pair_summary, plot_dir, cfg)
    plot_regime_specific_dashboard(pair_summary, plot_dir, cfg)
    plot_teacher_upper_bound(pair_summary, plot_dir, cfg)
    plot_regime_action(rows_out, plot_dir, cfg)
    plot_control_examples(rows_out, sample_payloads, plot_dir, cfg)

    # training curve
    hist = train_info.get("history", [])
    if hist:
        fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=cfg.dpi)
        ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist])
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Projection training loss")
        ax.grid(True, alpha=0.25)
        savefig(fig, os.path.join(plot_dir, "fig_training_loss"))

    print(json.dumps({"n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "pair_summary": pair_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

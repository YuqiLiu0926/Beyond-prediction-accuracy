# -*- coding: utf-8 -*-
"""continuation calibrationB: coarse-physics continuation selection and HF oracle viability audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from surrogate_pair_evaluator import Config, PairEvaluator


BASE = Path(__file__).resolve().parents[1]
PDES = ("burgers", "grayscott", "kolmogorov")


def parse_list(text: str, allowed: tuple[str, ...]) -> list[str]:
    values = [value.strip().lower() for value in text.split(",") if value.strip()]
    if not values or any(value not in allowed for value in values):
        raise ValueError(f"Expected a subset of {allowed}, got {text!r}")
    return values


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def candidate_library(horizon: int, umin: float, umax: float) -> tuple[list[str], np.ndarray]:
    if not (umin < 0.0 < umax):
        raise ValueError(f"Continuation library requires a signed control interval, got [{umin}, {umax}]")
    positive = float(umax)
    negative = float(umin)
    t = np.linspace(0.0, 1.0, horizon, dtype=np.float64)
    half = horizon // 2
    quarter = max(1, horizon // 4)
    names: list[str] = []
    values: list[np.ndarray] = []

    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        values.append(np.clip(np.asarray(value, dtype=np.float64), umin, umax))

    add("zero", np.zeros(horizon))
    for label, amplitude in (
        ("const_pos_half", 0.5 * positive),
        ("const_neg_half", 0.5 * negative),
        ("const_pos_full", positive),
        ("const_neg_full", negative),
    ):
        add(label, np.full(horizon, amplitude))
    for label, amplitude in (("pos", positive), ("neg", negative)):
        front = np.zeros(horizon); front[:half] = amplitude
        back = np.zeros(horizon); back[half:] = amplitude
        add(f"front_{label}", front)
        add(f"back_{label}", back)
        add(f"ramp_{label}", amplitude * t)
        triangle = amplitude * (1.0 - np.abs(2.0 * t - 1.0))
        add(f"triangle_{label}", triangle)
    alternating = np.empty(horizon, dtype=np.float64)
    for index in range(horizon):
        alternating[index] = positive if (index // quarter) % 2 == 0 else negative
    add("alternating_pos_first", alternating)
    add("alternating_neg_first", -alternating)
    return names, np.stack(values, axis=0)


def make_evaluator(args: argparse.Namespace, pde: str, horizon: int) -> PairEvaluator:
    cfg = Config()
    cfg.root = str(args.data_root)
    cfg.ckpt_root = str(args.ckpt_root)
    cfg.data_root = str(args.model_data_root)
    cfg.train_script = str(args.train_script)
    cfg.burgers_script = str(args.burgers_script)
    cfg.grayscott_script = str(args.grayscott_script)
    cfg.kolmogorov_script = str(args.kolmogorov_script)
    cfg.ic_mode = str(args.ic_mode)
    cfg.H = int(horizon)
    cfg.canonical_horizon = int(horizon)
    cfg.device = str(args.device)
    evaluator = PairEvaluator(pde, "deeponet", cfg)
    evaluator.exp_cfg.seed = int(args.ic_seed)
    evaluator.refresh_ics(args.n_ics, args.ic_mode)
    return evaluator


def physical_risk(evaluator: PairEvaluator, x0: np.ndarray, control: np.ndarray, fidelity: str, coarse_substeps: int) -> float:
    if fidelity not in {"coarse", "hf"}:
        raise KeyError(fidelity)
    if evaluator.pde == "burgers":
        if fidelity == "hf":
            state0 = evaluator.mod.periodic_interp_1d(x0, evaluator.x_nom, evaluator.x_true)
            grid = evaluator.x_true
            actuator = evaluator.k_true
            dx = evaluator.dx_true
            dt_true = float(evaluator.exp_cfg.dt_true)
        else:
            state0 = np.asarray(x0, dtype=np.float64)
            grid = evaluator.x_nom
            actuator = evaluator.k_nom
            dx = evaluator.dx_nom
            dt_true = float(evaluator.dt_nom) / coarse_substeps
        states = evaluator.mod.rollout_true(
            u0_true=state0,
            a_seq=control,
            x_true=grid,
            k_true=actuator,
            nu=evaluator.nu,
            dt_nom=evaluator.dt_nom,
            dt_true=dt_true,
        )
        return float(evaluator.mod.risk_Z_inf(states, dx))
    if evaluator.pde == "grayscott":
        dt_true = float(evaluator.exp_cfg.dt_true) if fidelity == "hf" else float(evaluator.dt_nom) / coarse_substeps
        states = evaluator.mod.rollout_true(
            s0_true=x0,
            a_seq=control,
            actuator=evaluator.actuator,
            dx=evaluator.dx,
            dy=evaluator.dy,
            Du=evaluator.Du,
            Dv=evaluator.Dv,
            F0=evaluator.F0,
            k0=evaluator.k0,
            dt_nom=evaluator.dt_nom,
            dt_true=dt_true,
        )
        return float(evaluator.mod.risk_Z_front(states, evaluator.dx, evaluator.dy))
    dt_true = float(evaluator.exp_cfg.dt_true) if fidelity == "hf" else float(evaluator.dt_nom) / coarse_substeps
    states = evaluator.mod.rollout_true(
        s0_true=x0,
        a_seq=control,
        actuator=evaluator.actuator,
        x=evaluator.x,
        y=evaluator.y,
        nu=evaluator.nu,
        dt_nom=evaluator.dt_nom,
        dt_true=dt_true,
        forcing_amp=evaluator.forcing_amp,
        forcing_k=evaluator.forcing_k,
    )
    return float(evaluator.mod.risk_Z_inf(states, evaluator.dx, evaluator.dy))


def summarize(rows: list[dict[str, Any]], scales: list[float], candidate_names: list[str]) -> dict[str, Any]:
    by_pde = {}
    for pde in PDES:
        local = [row for row in rows if row["pde"] == pde]
        if not local:
            continue
        oracle = np.asarray([min(row["z_hf_candidates"]) for row in local], dtype=np.float64)
        selected = np.asarray([row["z_hf_candidates"][row["lf_selected_index"]] for row in local], dtype=np.float64)
        zero = np.asarray([row["z_hf_candidates"][0] for row in local], dtype=np.float64)
        z0 = np.asarray([row["z0"] for row in local], dtype=np.float64)
        lf_flat = np.concatenate([np.asarray(row["z_lf_candidates"], dtype=np.float64) for row in local])
        hf_flat = np.concatenate([np.asarray(row["z_hf_candidates"], dtype=np.float64) for row in local])
        corr = spearmanr(lf_flat, hf_flat).statistic
        scale_rows = []
        for scale in scales:
            boundary = scale * z0
            scale_rows.append(
                {
                    "scale": scale,
                    "zero_safe": int(np.sum(zero <= boundary)),
                    "lf_selected_safe": int(np.sum(selected <= boundary)),
                    "oracle_library_safe": int(np.sum(oracle <= boundary)),
                    "selection_gap_to_oracle": int(np.sum(oracle <= boundary)) - int(np.sum(selected <= boundary)),
                }
            )
        selections = np.bincount(
            np.asarray([row["lf_selected_index"] for row in local], dtype=np.int64), minlength=len(candidate_names)
        )
        by_pde[pde] = {
            "n": len(local),
            "candidate_count": len(candidate_names),
            "lf_hf_spearman_all_candidates": float(corr),
            "median_oracle_ratio": float(np.median(oracle / z0)),
            "q95_oracle_ratio": float(np.quantile(oracle / z0, 0.95)),
            "median_lf_selected_ratio": float(np.median(selected / z0)),
            "q95_lf_selected_ratio": float(np.quantile(selected / z0, 0.95)),
            "median_selection_regret": float(np.median(selected - oracle)),
            "q95_selection_regret": float(np.quantile(selected - oracle, 0.95)),
            "selection_counts": {name: int(selections[i]) for i, name in enumerate(candidate_names)},
            "viability_by_scale": scale_rows,
        }
    return by_pde


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdes", default="burgers,grayscott,kolmogorov")
    parser.add_argument("--n-ics", type=int, default=100)
    parser.add_argument("--ic-seed", type=int, default=20260713)
    parser.add_argument("--ic-mode", default="ood_amplified")
    parser.add_argument("--coarse-substeps", type=int, default=2)
    parser.add_argument("--scales", default="1.02,1.05,1.10,1.20,1.50,2.00")
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=BASE / "external_data/decision_archives/detector_development")
    parser.add_argument("--outdir", type=Path, default=BASE / "outputs" / "continuation_calibration")
    parser.add_argument("--ckpt-root", type=Path, default=BASE / "external_data/checkpoints/discovery")
    parser.add_argument("--model-data-root", type=Path, default=BASE / "external_data/datasets/discovery")
    parser.add_argument("--train-script", type=Path, default=BASE / "scripts/train_discovery_surrogates.py")
    parser.add_argument("--burgers-script", type=Path, default=BASE / "scripts/generate_burgers_decisions.py")
    parser.add_argument("--grayscott-script", type=Path, default=BASE / "scripts/generate_gray_scott_decisions.py")
    parser.add_argument("--kolmogorov-script", type=Path, default=BASE / "scripts/generate_kolmogorov_decisions.py")
    args = parser.parse_args()
    if args.coarse_substeps < 1:
        raise ValueError("coarse-substeps must be positive")

    pdes = parse_list(args.pdes, PDES)
    scales = [float(value) for value in args.scales.split(",") if value.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / "continuation_candidate_rows.jsonl"
    if not int(args.resume) and output.exists():
        output.unlink()
    done = {(row["pde"], int(row["ic_idx"])) for row in iter_jsonl(output)}

    candidate_names = None
    started = time.perf_counter()
    for pde in pdes:
        path = args.data_root / pde / "deeponet" / args.ic_mode / "decision_data.npz"
        with np.load(path, allow_pickle=True) as data:
            states = np.asarray(data["initial_states"][: args.n_ics], dtype=np.float64)
            z0s = np.asarray(data["z0"][: args.n_ics], dtype=np.float64)
            horizon = int(np.asarray(data["U_all"]).shape[1])
        evaluator = make_evaluator(args, pde, horizon)
        names, candidates = candidate_library(horizon, evaluator.umin, evaluator.umax)
        if candidate_names is None:
            candidate_names = names
        elif names != candidate_names:
            raise RuntimeError("Candidate names changed across PDEs")
        for ic_idx, (x0, z0) in enumerate(zip(states, z0s)):
            if (pde, ic_idx) in done:
                continue
            regenerated = np.asarray(evaluator.x0s[ic_idx], dtype=np.float64)
            state_error = float(np.max(np.abs(regenerated - x0)))
            if state_error > 1e-6:
                raise RuntimeError(f"Initial-state mismatch for {pde}:{ic_idx}: {state_error}")
            lf_started = time.perf_counter()
            z_lf = [physical_risk(evaluator, x0, control, "coarse", args.coarse_substeps) for control in candidates]
            lf_seconds = time.perf_counter() - lf_started
            hf_started = time.perf_counter()
            z_hf = [physical_risk(evaluator, x0, control, "hf", args.coarse_substeps) for control in candidates]
            hf_seconds = time.perf_counter() - hf_started
            if not np.all(np.isfinite(z_lf)) or not np.all(np.isfinite(z_hf)):
                raise RuntimeError(f"Non-finite continuation risk for {pde}:{ic_idx}")
            row = {
                "pde": pde,
                "ic_idx": ic_idx,
                "z0": float(z0),
                "candidate_names": names,
                "z_lf_candidates": [float(value) for value in z_lf],
                "z_hf_candidates": [float(value) for value in z_hf],
                "lf_selected_index": int(np.argmin(z_lf)),
                "hf_oracle_index": int(np.argmin(z_hf)),
                "lf_seconds": lf_seconds,
                "hf_seconds": hf_seconds,
                "initial_state_max_abs_error_to_regenerated": state_error,
            }
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"[continuation] {pde} {ic_idx + 1}/{len(states)} "
                f"lf={names[row['lf_selected_index']]} oracle={names[row['hf_oracle_index']]} "
                f"ratio={min(z_hf) / z0:.4f}",
                flush=True,
            )

    rows = list(iter_jsonl(output))
    expected = {(pde, i) for pde in pdes for i in range(args.n_ics)}
    keys = {(str(row["pde"]), int(row["ic_idx"])) for row in rows if row["pde"] in pdes and int(row["ic_idx"]) < args.n_ics}
    if keys != expected:
        raise RuntimeError(f"Incomplete continuation audit: observed={len(keys)} expected={len(expected)}")
    selected_rows = [row for row in rows if (row["pde"], int(row["ic_idx"])) in expected]
    summary = {
        "protocol": "continuation calibrationB fixed-library continuation viability",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "candidate_names": candidate_names,
        "candidate_count": len(candidate_names or []),
        "coarse_solver": {
            "known_PDE_operator": True,
            "coarse_time_substeps_per_nominal_step": args.coarse_substeps,
            "surrogate_model_used_for_selection": False,
            "online_HF_used_for_selection": False,
        },
        "scales": scales,
        "by_pde": summarize(selected_rows, scales, candidate_names or []),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": len(selected_rows),
        "records": str(output),
    }
    summary_path = args.outdir / f"continuation_viability_n{args.n_ics}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    table_rows = []
    for pde, local in summary["by_pde"].items():
        for row in local["viability_by_scale"]:
            table_rows.append({"pde": pde, **row})
    with (args.outdir / f"continuation_viability_n{args.n_ics}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader(); writer.writerows(table_rows)
    print(json.dumps({"status": "PASS", "summary": str(summary_path), "rows": len(selected_rows)}))


if __name__ == "__main__":
    main()

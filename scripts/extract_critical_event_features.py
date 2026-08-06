"""Extract aggregate and decision-defining critical-event fidelity metrics.

The experiment freezes the 2,400 optimized controls from the discovery decision study, replays each
control through the matching surrogate and HF solver, and reduces the two
state trajectories to scalar diagnostics. Full trajectories are not retained.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from operator_defect_features import (
    surrogate_metrics_from_states,
    surrogate_states,
)
from surrogate_pair_evaluator import Config, PairEvaluator


BASE = Path(__file__).resolve().parents[1]
PDES = ("burgers", "grayscott", "kolmogorov")
MODELS = ("deeponet", "fno", "pino", "pinn")
TAIL_QUANTILES = (0.90, 0.95, 0.99)


def parse_csv(text: str) -> list[str]:
    return [item.strip().lower() for item in str(text).split(",") if item.strip()]


def parse_pairs(text: str) -> list[tuple[str, str]]:
    if not str(text).strip():
        return [(pde, model) for pde in PDES for model in MODELS]
    pairs: list[tuple[str, str]] = []
    for token in parse_csv(text):
        if ":" not in token:
            raise ValueError(f"Pair must be pde:model, got {token!r}")
        pde, model = token.split(":", 1)
        if pde not in PDES or model not in MODELS:
            raise ValueError(f"Unknown pair {token!r}")
        pairs.append((pde, model))
    return pairs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    return {
        (str(row["pde"]), str(row["model"]), int(row["ic_idx"]))
        for row in read_jsonl(path)
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_cfg(args: argparse.Namespace, horizon: int) -> Config:
    cfg = Config()
    cfg.root = str(args.cache_root)
    cfg.ckpt_root = str(args.ckpt_root)
    cfg.data_root = str(args.data_root)
    cfg.train_script = str(args.train_script)
    cfg.burgers_script = str(args.burgers_script)
    cfg.grayscott_script = str(args.grayscott_script)
    cfg.kolmogorov_script = str(args.kolmogorov_script)
    cfg.ic_mode = "challenge_conditions"
    cfg.H = int(horizon)
    cfg.canonical_horizon = int(horizon)
    cfg.reevaluate_baseline = 0
    cfg.device = str(args.device)
    return cfg


def configure_determinism(seed: int) -> None:
    np.random.seed(int(seed))
    if torch is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def normalize_surrogate_states(pde: str, states: np.ndarray, horizon: int) -> np.ndarray:
    array = np.asarray(states, dtype=np.float64)
    if array.ndim >= 2 and array.shape[0] == 1 and array.shape[1] == horizon + 1:
        array = array[0]
    if pde == "burgers" and array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    return array


def hf_states_and_metrics(
    evaluator: PairEvaluator, x0: np.ndarray, control: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    control = np.asarray(control, dtype=np.float64).ravel()
    if evaluator.pde == "burgers":
        x0_true = evaluator.mod.periodic_interp_1d(x0, evaluator.x_nom, evaluator.x_true)
        states = evaluator.mod.rollout_true(
            u0_true=x0_true,
            a_seq=control,
            x_true=evaluator.x_true,
            k_true=evaluator.k_true,
            nu=evaluator.nu,
            dt_nom=evaluator.dt_nom,
            dt_true=float(evaluator.exp_cfg.dt_true),
        )
        z_ts = evaluator.mod.risk_inf_series(states, evaluator.dx_true)
        j_ts = evaluator.mod.response_series_1d(states, evaluator.k_true)
        z_value = evaluator.mod.risk_Z_inf(states, evaluator.dx_true)
        j_value = evaluator.mod.terminal_response(states[-1], evaluator.k_true)
    elif evaluator.pde == "grayscott":
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
            dt_true=float(evaluator.exp_cfg.dt_true),
        )
        z_ts = evaluator.mod.risk_front_series(states, evaluator.dx, evaluator.dy)
        j_ts = evaluator.mod.response_series(states, evaluator.actuator)
        z_value = evaluator.mod.risk_Z_front(states, evaluator.dx, evaluator.dy)
        j_value = evaluator.mod.nominal_performance(states, evaluator.actuator)
    else:
        states = evaluator.mod.rollout_true(
            s0_true=x0,
            a_seq=control,
            actuator=evaluator.actuator,
            x=evaluator.x,
            y=evaluator.y,
            nu=evaluator.nu,
            dt_nom=evaluator.dt_nom,
            dt_true=float(evaluator.exp_cfg.dt_true),
            forcing_amp=evaluator.forcing_amp,
            forcing_k=evaluator.forcing_k,
        )
        z_ts = evaluator.mod.risk_inf_series(states, evaluator.dx, evaluator.dy)
        j_ts = evaluator.mod.response_series(states, evaluator.actuator)
        z_value = evaluator.mod.risk_Z_inf(states, evaluator.dx, evaluator.dy)
        j_value = evaluator.mod.nominal_performance(states, evaluator.actuator)
    return np.asarray(states, dtype=np.float64), {
        "z_hf": float(z_value),
        "J_hf": float(j_value),
        "z_hf_ts": np.asarray(z_ts, dtype=np.float64),
        "J_hf_ts": np.asarray(j_ts, dtype=np.float64),
    }


def common_grid_states(
    evaluator: PairEvaluator, states_sur: np.ndarray, states_hf: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    states_sur = np.asarray(states_sur, dtype=np.float64)
    states_hf = np.asarray(states_hf, dtype=np.float64)
    if evaluator.pde == "burgers":
        states_hf = evaluator.mod.interp_true_states_to_nom(
            states_hf, evaluator.x_true, evaluator.x_nom
        )
    if states_sur.shape != states_hf.shape:
        raise ValueError(
            f"Common-grid shape mismatch for {evaluator.pde}: "
            f"sur={states_sur.shape}, hf={states_hf.shape}"
        )
    return states_sur, states_hf


def risk_channel(pde: str, states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if pde == "burgers":
        return states
    if pde == "grayscott":
        return states[:, 1, :, :]
    return states[:, 0, :, :]


def risk_density(evaluator: PairEvaluator, states: np.ndarray) -> np.ndarray:
    channel = risk_channel(evaluator.pde, states)
    if evaluator.pde == "burgers":
        return np.abs(
            (np.roll(channel, -1, axis=-1) - np.roll(channel, 1, axis=-1))
            / (2.0 * float(evaluator.dx_nom))
        )
    gx = (
        np.roll(channel, -1, axis=-1) - np.roll(channel, 1, axis=-1)
    ) / (2.0 * float(evaluator.dx))
    gy = (
        np.roll(channel, -1, axis=-2) - np.roll(channel, 1, axis=-2)
    ) / (2.0 * float(evaluator.dy))
    return np.sqrt(gx * gx + gy * gy)


def relative_l2(surrogate: np.ndarray, reference: np.ndarray) -> float:
    surrogate = np.asarray(surrogate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    numerator = float(np.sum((surrogate - reference) ** 2))
    denominator = float(np.sum(reference**2))
    return float(math.sqrt(numerator / max(denominator, 1e-30)))


def per_time_relative_l2(surrogate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    surrogate = np.asarray(surrogate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    axes = tuple(range(1, surrogate.ndim))
    numerator = np.sum((surrogate - reference) ** 2, axis=axes)
    denominator = np.sum(reference**2, axis=axes)
    return np.sqrt(numerator / np.maximum(denominator, 1e-30))


def periodic_spatial_displacement(
    pde: str, index_sur: tuple[int, ...], index_hf: tuple[int, ...], shape: tuple[int, ...]
) -> float:
    if pde == "burgers":
        n = int(shape[-1])
        delta = abs(int(index_sur[-1]) - int(index_hf[-1]))
        return float(min(delta, n - delta) / max(1, n))
    ny, nx = int(shape[-2]), int(shape[-1])
    dy = abs(int(index_sur[-2]) - int(index_hf[-2]))
    dx = abs(int(index_sur[-1]) - int(index_hf[-1]))
    dy = min(dy, ny - dy) / max(1, ny)
    dx = min(dx, nx - dx) / max(1, nx)
    return float(math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0))


def trajectory_features(
    evaluator: PairEvaluator,
    states_sur: np.ndarray,
    states_hf: np.ndarray,
    z_sur_ts_native: np.ndarray,
    z_hf_ts_native: np.ndarray,
    z_limit: float,
) -> dict[str, float]:
    states_sur, states_hf = common_grid_states(evaluator, states_sur, states_hf)
    channel_sur = risk_channel(evaluator.pde, states_sur)
    channel_hf = risk_channel(evaluator.pde, states_hf)
    density_sur = risk_density(evaluator, states_sur)
    density_hf = risk_density(evaluator, states_hf)
    density_error = density_sur - density_hf
    scale = max(abs(float(z_limit)), 1e-12)

    rel_time = per_time_relative_l2(states_sur, states_hf)
    channel_rel_time = per_time_relative_l2(channel_sur, channel_hf)
    global_density_rms = float(np.sqrt(np.mean(density_error**2)) / scale)

    out: dict[str, float] = {
        "field_rel_l2_all": relative_l2(states_sur, states_hf),
        "field_rel_l2_final": relative_l2(states_sur[-1], states_hf[-1]),
        "field_rel_l2_time_mean": float(np.mean(rel_time)),
        "field_rel_l2_time_max": float(np.max(rel_time)),
        "risk_channel_rel_l2_all": relative_l2(channel_sur, channel_hf),
        "risk_channel_rel_l2_final": relative_l2(channel_sur[-1], channel_hf[-1]),
        "risk_channel_rel_l2_time_mean": float(np.mean(channel_rel_time)),
        "risk_density_error_rms_global_norm": global_density_rms,
    }

    flat_hf = density_hf.ravel()
    flat_error = density_error.ravel()
    for quantile in TAIL_QUANTILES:
        threshold = float(np.quantile(flat_hf, quantile))
        mask = flat_hf >= threshold
        tail_rms = float(np.sqrt(np.mean(flat_error[mask] ** 2)) / scale)
        signed_under = float(np.mean(-flat_error[mask]) / scale)
        name = int(round(100 * quantile))
        out[f"risk_density_error_rms_q{name}_norm"] = tail_rms
        out[f"risk_density_concentration_q{name}"] = float(
            tail_rms / max(global_density_rms, 1e-12)
        )
        out[f"risk_density_underestimate_mean_q{name}_norm"] = signed_under

    hf_flat_idx = int(np.argmax(density_hf))
    sur_flat_idx = int(np.argmax(density_sur))
    hf_idx = np.unravel_index(hf_flat_idx, density_hf.shape)
    sur_idx = np.unravel_index(sur_flat_idx, density_sur.shape)
    t_hf = int(hf_idx[0])
    t_sur = int(sur_idx[0])
    hf_critical = float(density_hf[hf_idx])
    sur_at_hf = float(density_sur[hf_idx])
    sur_peak = float(density_sur[sur_idx])
    hf_at_sur = float(density_hf[sur_idx])
    horizon = max(1, density_hf.shape[0] - 1)
    out.update(
        {
            "critical_hf_density_norm": hf_critical / scale,
            "critical_sur_density_at_hf_norm": sur_at_hf / scale,
            "critical_signed_underestimate_norm": (hf_critical - sur_at_hf) / scale,
            "critical_abs_error_norm": abs(hf_critical - sur_at_hf) / scale,
            "surrogate_peak_density_norm": sur_peak / scale,
            "hf_density_at_surrogate_peak_norm": hf_at_sur / scale,
            "common_grid_peak_gap_norm": (hf_critical - sur_peak) / scale,
            "peak_time_hf_frac": t_hf / horizon,
            "peak_time_sur_frac": t_sur / horizon,
            "peak_time_displacement_frac": abs(t_hf - t_sur) / horizon,
            "peak_space_displacement_frac": periodic_spatial_displacement(
                evaluator.pde, sur_idx, hf_idx, density_hf.shape
            ),
        }
    )

    z_sur_ts_native = np.asarray(z_sur_ts_native, dtype=np.float64)
    z_hf_ts_native = np.asarray(z_hf_ts_native, dtype=np.float64)
    trace_error = z_hf_ts_native - z_sur_ts_native
    out.update(
        {
            "risk_trace_mae_norm": float(np.mean(np.abs(trace_error)) / scale),
            "risk_trace_rmse_norm": float(np.sqrt(np.mean(trace_error**2)) / scale),
            "risk_trace_bias_hf_minus_sur_norm": float(np.mean(trace_error) / scale),
            "risk_trace_max_abs_error_norm": float(np.max(np.abs(trace_error)) / scale),
            "label_defining_peak_gap_norm": float(
                (np.max(z_hf_ts_native) - np.max(z_sur_ts_native)) / scale
            ),
        }
    )
    for quantile in (0.50, 0.90, 0.95, 0.99):
        name = int(round(100 * quantile))
        out[f"risk_trace_quantile_gap_q{name}_norm"] = float(
            (np.quantile(z_hf_ts_native, quantile) - np.quantile(z_sur_ts_native, quantile))
            / scale
        )
    return out


def regime(m_sur: float, m_hf: float) -> str:
    if m_sur >= 0.0 and m_hf >= 0.0:
        return "safe_agreement"
    if m_sur >= 0.0 and m_hf < 0.0:
        return "false_safe"
    if m_sur < 0.0 and m_hf >= 0.0:
        return "conservative_rejection"
    return "unsafe_agreement"


def load_cache_metadata(cache_root: Path, pde: str, model: str) -> dict[str, np.ndarray]:
    path = cache_root / pde / model / "challenge_conditions" / "decision_data.npz"
    with np.load(path, allow_pickle=True) as data:
        return {
            "tags": np.asarray(data["tags"], object),
            "margin_sur": np.asarray(data["margin_sur"], dtype=np.float64),
            "margin_hf": np.asarray(data["margin_hf"], dtype=np.float64),
            "z_sur": np.asarray(data["z_sur"], dtype=np.float64),
            "z_hf": np.asarray(data["z_hf"], dtype=np.float64),
            "U_all": np.asarray(data["U_all"], dtype=np.float64),
        }


def write_manifest(
    args: argparse.Namespace, output_dir: Path, pairs: list[tuple[str, str]], source_rows: int
) -> None:
    manifest = {
        "analysis": "Critical-event fidelity under matched aggregate accuracy",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_jsonl": str(args.replay_jsonl.resolve()),
        "source_jsonl_sha256": sha256(args.replay_jsonl),
        "cache_root": str(args.cache_root.resolve()),
        "pairs": [f"{pde}:{model}" for pde, model in pairs],
        "source_rows": int(source_rows),
        "device": str(args.device),
        "tail_quantiles": list(TAIL_QUANTILES),
        "label_source": "fresh deterministic replay",
        "full_state_trajectories_saved": False,
        "command_arguments": {key: str(value) for key, value in vars(args).items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> Path:
    configure_determinism(int(args.seed))
    pairs = parse_pairs(args.pairs)
    selected = set(pairs)
    source_rows = read_jsonl(args.replay_jsonl)
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        key = (str(row["pde"]), str(row["model"]))
        if key in selected:
            by_pair[key].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "critical_event_features.jsonl"
    if not int(args.resume) and output_path.exists():
        output_path.unlink()
    done = completed_keys(output_path) if int(args.resume) else set()
    write_manifest(args, args.output_dir, pairs, len(source_rows))

    for pde, model in pairs:
        rows = sorted(by_pair[(pde, model)], key=lambda item: int(item["ic_idx"]))
        if int(args.max_controls_per_pair) > 0:
            rows = rows[: int(args.max_controls_per_pair)]
        if not rows:
            raise RuntimeError(f"No replay rows found for {pde}:{model}")
        horizon = int(rows[0]["horizon"])
        print(f"[critical-event extraction] {pde}:{model} n={len(rows)} H={horizon}", flush=True)
        evaluator = PairEvaluator(pde, model, make_cfg(args, horizon))
        if torch is not None and str(args.device).startswith("cuda") and torch.cuda.is_available():
            evaluator.model.model.to(torch.device(args.device))
        evaluator.refresh_ics(200, "challenge_conditions")
        cache = load_cache_metadata(args.cache_root, pde, model)

        for position, source in enumerate(rows, start=1):
            ic_idx = int(source["ic_idx"])
            key = (pde, model, ic_idx)
            if key in done:
                continue
            started = time.perf_counter()
            control = np.asarray(source["U"], dtype=np.float64).ravel()
            cache_control_error = float(np.max(np.abs(control - cache["U_all"][ic_idx])))
            if cache_control_error > 1e-7:
                raise RuntimeError(
                    f"Control mismatch for {key}: max abs error={cache_control_error}"
                )
            x0 = np.asarray(evaluator.x0s[ic_idx], dtype=np.float64)
            states_sur = normalize_surrogate_states(
                pde, surrogate_states(evaluator, x0, control), horizon
            )
            sur_metrics = surrogate_metrics_from_states(evaluator, states_sur)
            states_hf, hf_metrics = hf_states_and_metrics(evaluator, x0, control)
            z_limit = float(source["z_limit"])
            m_sur = z_limit - float(sur_metrics["z_sur_replay"])
            m_hf = z_limit - float(hf_metrics["z_hf"])
            features = trajectory_features(
                evaluator,
                states_sur,
                states_hf,
                sur_metrics["z_sur_ts_replay"],
                hf_metrics["z_hf_ts"],
                z_limit,
            )
            z_sur_replay_error = abs(
                float(sur_metrics["z_sur_replay"]) - float(source["z_sur_replay"])
            )
            z_hf_replay_error = abs(float(hf_metrics["z_hf"]) - float(source["z_hf_replay"]))
            if z_hf_replay_error > float(args.hf_replay_tolerance):
                raise RuntimeError(
                    f"HF replay mismatch for {key}: error={z_hf_replay_error}"
                )
            if z_sur_replay_error > float(args.surrogate_replay_fail_tolerance):
                raise RuntimeError(
                    f"Surrogate replay divergence for {key}: error={z_sur_replay_error}"
                )

            cache_m_sur = float(cache["margin_sur"][ic_idx])
            cache_m_hf = float(cache["margin_hf"][ic_idx])
            saved_m_sur = float(source["m_sur_replay"])
            saved_m_hf = float(source["m_hf_replay"])
            output = {
                "pde": pde,
                "model": model,
                "pair": f"{pde}:{model}",
                "ic_idx": ic_idx,
                "physical_group": f"{pde}:{ic_idx}",
                "tag": str(cache["tags"][ic_idx]),
                "horizon": horizon,
                "z_limit": z_limit,
                "z_sur": float(sur_metrics["z_sur_replay"]),
                "z_hf": float(hf_metrics["z_hf"]),
                "m_sur": m_sur,
                "m_hf": m_hf,
                "m_sur_norm": m_sur / max(abs(z_limit), 1e-12),
                "m_hf_norm": m_hf / max(abs(z_limit), 1e-12),
                "regime": regime(m_sur, m_hf),
                "false_safe": int(m_sur >= 0.0 and m_hf < 0.0),
                "surrogate_accepted": int(m_sur >= 0.0),
                "cache_regime": regime(cache_m_sur, cache_m_hf),
                "cache_m_sur": cache_m_sur,
                "cache_m_hf": cache_m_hf,
                "cache_to_replay_regime_changed": int(
                    regime(cache_m_sur, cache_m_hf) != regime(m_sur, m_hf)
                ),
                "saved_replay_regime": regime(saved_m_sur, saved_m_hf),
                "saved_replay_m_sur": saved_m_sur,
                "saved_replay_m_hf": saved_m_hf,
                "saved_to_fresh_regime_changed": int(
                    regime(saved_m_sur, saved_m_hf) != regime(m_sur, m_hf)
                ),
                "z_sur_abs_error_to_saved_replay": z_sur_replay_error,
                "z_hf_abs_error_to_saved_replay": z_hf_replay_error,
                "control_max_abs_error_to_cache": cache_control_error,
                "control_mean": float(np.mean(control)),
                "control_std": float(np.std(control)),
                "control_l2": float(np.sqrt(np.mean(control**2))),
                "control_tv": float(np.sum(np.abs(np.diff(control)))),
                "runtime_sec": float(time.perf_counter() - started),
                **features,
            }
            append_jsonl(output_path, output)
            done.add(key)
            if position % max(1, int(args.progress_every)) == 0 or position == len(rows):
                print(
                    f"  {position:4d}/{len(rows)} ic={ic_idx:3d} "
                    f"regime={output['regime']} l2={output['field_rel_l2_all']:.4g} "
                    f"tail={output['risk_density_concentration_q99']:.3g} "
                    f"t={output['runtime_sec']:.2f}s",
                    flush=True,
                )

        del evaluator
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows_written = read_jsonl(output_path)
    summary = {
        "rows": len(rows_written),
        "unique_keys": len(
            {(row["pde"], row["model"], int(row["ic_idx"])) for row in rows_written}
        ),
        "false_safe": int(sum(int(row["false_safe"]) for row in rows_written)),
        "surrogate_accepted": int(
            sum(int(row["surrogate_accepted"]) for row in rows_written)
        ),
        "cache_to_replay_regime_changes": int(
            sum(int(row["cache_to_replay_regime_changed"]) for row in rows_written)
        ),
        "saved_to_fresh_regime_changes": int(
            sum(int(row["saved_to_fresh_regime_changed"]) for row in rows_written)
        ),
        "max_z_sur_replay_error": float(
            max(row["z_sur_abs_error_to_saved_replay"] for row in rows_written)
        ),
        "max_z_hf_replay_error": float(
            max(row["z_hf_abs_error_to_saved_replay"] for row in rows_written)
        ),
        "total_runtime_sec": float(sum(row["runtime_sec"] for row in rows_written)),
    }
    (args.output_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-jsonl",
        type=Path,
        default=BASE
        / "external_data"
        / "analysis"
        / "risk_geometry"
        / "decision_features.jsonl",
    )
    parser.add_argument("--cache-root", type=Path, default=BASE / "external_data/decision_archives/discovery")
    parser.add_argument(
        "--ckpt-root", type=Path, default=BASE / "external_data" / "checkpoints"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=BASE / "external_data" / "datasets" / "discovery",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=BASE / "scripts" / "train_discovery_surrogates.py",
    )
    parser.add_argument(
        "--burgers-script",
        type=Path,
        default=BASE / "scripts/generate_burgers_decisions.py",
    )
    parser.add_argument(
        "--grayscott-script",
        type=Path,
        default=BASE / "scripts/generate_gray_scott_decisions.py",
    )
    parser.add_argument(
        "--kolmogorov-script",
        type=Path,
        default=BASE / "scripts/generate_kolmogorov_decisions.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "outputs" / "critical_event_analysis"
    )
    parser.add_argument("--pairs", default="")
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-controls-per-pair", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--hf-replay-tolerance", type=float, default=5e-4)
    parser.add_argument("--surrogate-replay-fail-tolerance", type=float, default=10.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

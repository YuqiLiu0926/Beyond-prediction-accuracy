"""Rebuild missing tabular metadata for staged decision archives.

The archived NPZ files contain the scientific arrays and embedded run metadata.
This utility creates a flat decision table, a summary, and a run manifest only
when one of those companion files is absent. Quantities that were not stored in
the NPZ archive, such as per-iteration optimizer diagnostics, are left empty.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


TABLE_FIELDS = [
    "run_id",
    "data_split",
    "pde",
    "model_type",
    "condition_set",
    "condition_index",
    "condition_type",
    "initial_risk",
    "surrogate_risk",
    "hf_risk",
    "hf_minus_surrogate_risk",
    "safety_limit",
    "surrogate_margin",
    "hf_margin",
    "surrogate_excess",
    "hf_excess",
    "surrogate_task",
    "hf_task",
    "hf_minus_surrogate_task",
    "risk_underestimated",
    "surrogate_safe",
    "hf_safe",
    "false_safe",
    "control_mean",
    "control_std",
    "control_min",
    "control_max",
    "control_total_variation",
    "block_control_mean",
    "block_control_std",
    "optimizer_reference_risk",
    "optimizer_nonlinear_risk",
    "optimizer_reference_task",
    "optimizer_nonlinear_task",
    "accepted_step_size",
    "optimizer_step_accepted",
    "backtracking_trials",
    "rejected_steps",
    "optimizer_iterations",
]


def scalar_text(value: Any) -> str:
    """Convert a stored scalar to a JSON-safe string."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def public_pde(value: str) -> str:
    key = value.lower().replace("-", "_")
    if key in {"grayscott", "gray_scott"}:
        return "gray_scott"
    return key


def public_condition(value: str) -> str:
    return {
        "high_grad": "high_gradient",
        "ood_amp": "amplified_out_of_distribution",
    }.get(value, value)


def load_embedded_json(archive: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    if key not in archive.files or archive[key].size == 0:
        return {}
    raw = scalar_text(archive[key].reshape(-1)[0])
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def vector(archive: np.lib.npyio.NpzFile, key: str, n: int) -> np.ndarray:
    if key not in archive.files:
        return np.full(n, np.nan, dtype=float)
    values = np.asarray(archive[key]).reshape(-1)
    if len(values) != n:
        raise ValueError(f"{key} has {len(values)} entries; expected {n}")
    return values


def optional_value(values: np.ndarray, index: int) -> str | float | int:
    if index >= len(values):
        return ""
    value = values[index]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def archive_identity(path: Path, data_root: Path) -> tuple[str, str, str]:
    relative = path.relative_to(data_root)
    if len(relative.parts) < 6 or relative.parts[0] != "decision_archives":
        raise ValueError(f"Unexpected decision archive path: {relative}")
    split, pde, model = relative.parts[1:4]
    return split, public_pde(pde), model.lower()


def build_table(
    archive: np.lib.npyio.NpzFile,
    split: str,
    pde: str,
    model: str,
    run_id: str,
) -> list[dict[str, Any]]:
    n = len(np.asarray(archive["z_hf"]).reshape(-1))
    arrays = {
        key: vector(archive, key, n)
        for key in (
            "z0",
            "z_sur",
            "z_hf",
            "gap",
            "z_limit",
            "margin_sur",
            "margin_hf",
            "surrogate_excess",
            "hf_excess",
            "J_sur",
            "J_hf",
        )
    }
    tags = np.asarray(archive["tags"], dtype=object).reshape(-1)
    controls = np.asarray(archive["U_all"], dtype=float)
    block_controls = np.asarray(archive["alpha_all"], dtype=float)
    if controls.shape[0] != n or block_controls.shape[0] != n:
        raise ValueError("Control arrays do not match the number of decisions")

    rows: list[dict[str, Any]] = []
    for index in range(n):
        u = controls[index].reshape(-1)
        alpha = block_controls[index].reshape(-1)
        z_sur = float(arrays["z_sur"][index])
        z_hf = float(arrays["z_hf"][index])
        margin_sur = float(arrays["margin_sur"][index])
        margin_hf = float(arrays["margin_hf"][index])
        row = {
            "run_id": run_id,
            "data_split": split,
            "pde": pde,
            "model_type": model,
            "condition_set": "challenge_conditions",
            "condition_index": index,
            "condition_type": public_condition(scalar_text(tags[index])),
            "initial_risk": optional_value(arrays["z0"], index),
            "surrogate_risk": z_sur,
            "hf_risk": z_hf,
            "hf_minus_surrogate_risk": float(arrays["gap"][index]),
            "safety_limit": optional_value(arrays["z_limit"], index),
            "surrogate_margin": margin_sur,
            "hf_margin": margin_hf,
            "surrogate_excess": optional_value(arrays["surrogate_excess"], index),
            "hf_excess": optional_value(arrays["hf_excess"], index),
            "surrogate_task": optional_value(arrays["J_sur"], index),
            "hf_task": optional_value(arrays["J_hf"], index),
            "hf_minus_surrogate_task": float(arrays["J_hf"][index] - arrays["J_sur"][index]),
            "risk_underestimated": int(z_hf > z_sur),
            "surrogate_safe": int(margin_sur >= 0.0),
            "hf_safe": int(margin_hf >= 0.0),
            "false_safe": int(margin_sur >= 0.0 and margin_hf < 0.0),
            "control_mean": float(np.mean(u)),
            "control_std": float(np.std(u)),
            "control_min": float(np.min(u)),
            "control_max": float(np.max(u)),
            "control_total_variation": float(np.sum(np.abs(np.diff(u)))),
            "block_control_mean": float(np.mean(alpha)),
            "block_control_std": float(np.std(alpha)),
        }
        for field in TABLE_FIELDS:
            row.setdefault(field, "")
        rows.append(row)
    return rows


def public_summary(
    embedded: dict[str, Any],
    config: dict[str, Any],
    folder: Path,
    data_root: Path,
    split: str,
    pde: str,
    model: str,
    n: int,
) -> dict[str, Any]:
    payload = dict(embedded)
    run_id = f"{split}_{pde}_{model}_challenge_conditions"
    payload["experiment_name"] = "surrogate_guided_decision_evaluation"
    payload["run_id"] = run_id
    payload.pop("timestamp", None)
    payload["pde"] = pde
    payload["model_type"] = model
    payload["ic_mode"] = "challenge_conditions"
    payload["n_eval_ic"] = n
    payload["ckpt_path"] = f"checkpoints/discovery/{pde}/{model}.pt"
    payload["dataset_path"] = f"datasets/discovery/{pde}_controlled_trajectories.npz"
    payload["outdir"] = folder.relative_to(data_root).as_posix()
    cfg = dict(payload.get("cfg") if isinstance(payload.get("cfg"), dict) else config)
    cfg.update(
        {
            "train_script": "scripts/train_discovery_surrogates.py",
            "ckpt_path": payload["ckpt_path"],
            "dataset_path": payload["dataset_path"],
            "outdir": payload["outdir"],
            "ic_mode": "challenge_conditions",
        }
    )
    payload["cfg"] = cfg
    metadata = payload.get("checkpoint_metadata")
    if isinstance(metadata, dict):
        metadata["train_script"] = "scripts/train_discovery_surrogates.py"
    return payload


def public_manifest(
    folder: Path,
    data_root: Path,
    split: str,
    pde: str,
    model: str,
    n: int,
    seed: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "data_split": split,
        "pde": pde,
        "surrogate": model,
        "condition_set": "challenge_conditions",
        "n_decisions": n,
        "random_seed": seed,
        "files": {
            "summary": (folder / "summary.json").relative_to(data_root).as_posix(),
            "decision_table": (folder / "decision_summary.csv").relative_to(data_root).as_posix(),
            "arrays": (folder / "decision_data.npz").relative_to(data_root).as_posix(),
        },
        "checkpoint": f"checkpoints/discovery/{pde}/{model}.pt",
        "dataset": f"datasets/discovery/{pde}_controlled_trajectories.npz",
        "generation_script": f"scripts/generate_{pde}_decisions.py",
    }


def complete_archive(path: Path, data_root: Path, overwrite: bool) -> bool:
    folder = path.parent
    outputs = [folder / "decision_summary.csv", folder / "summary.json", folder / "run_manifest.json"]
    if not overwrite and all(output.exists() for output in outputs):
        return False

    split, pde, model = archive_identity(path, data_root)
    run_id = f"{split}_{pde}_{model}_challenge_conditions"
    with np.load(path, allow_pickle=True) as archive:
        embedded = load_embedded_json(archive, "summary")
        config = load_embedded_json(archive, "config")
        rows = build_table(archive, split, pde, model, run_id)
    summary = public_summary(embedded, config, folder, data_root, split, pde, model, len(rows))
    seed = summary.get("cfg", {}).get("seed")
    manifest = public_manifest(folder, data_root, split, pde, model, len(rows), seed)

    if overwrite or not outputs[0].exists():
        with outputs[0].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    if overwrite or not outputs[1].exists():
        outputs[1].write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if overwrite or not outputs[2].exists():
        outputs[2].write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "neural-pde-decision-reliability-data",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    rebuilt = []
    for path in sorted((data_root / "decision_archives").rglob("decision_data.npz")):
        if complete_archive(path, data_root, args.overwrite):
            rebuilt.append(path.parent.relative_to(data_root).as_posix())
    print(json.dumps({"rebuilt": len(rebuilt), "archives": rebuilt}, indent=2))


if __name__ == "__main__":
    main()

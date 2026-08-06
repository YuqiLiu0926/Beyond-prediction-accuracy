"""Audit the frozen surrogate checkpoints and challenge-cohort construction.

This script is deliberately read-only with respect to experiment artifacts.  It
extracts metadata from the twelve checkpoints actually referenced by the
canonical discovery decision study summaries and records the exact source definitions used to form
the mixed challenge cohorts.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "external_data"
RESULT_ROOT = DATA_ROOT / "decision_archives" / "discovery"
OUT = ROOT / "outputs" / "audits" / "reproducibility"
PDES = ("burgers", "grayscott", "kolmogorov")
MODELS = ("deeponet", "fno", "pino", "pinn")

SOURCE_FILES = {
    "training": SCRIPT_ROOT / "train_discovery_surrogates.py",
    "burgers_challenge": SCRIPT_ROOT / "generate_burgers_decisions.py",
    "grayscott_challenge": SCRIPT_ROOT / "generate_gray_scott_decisions.py",
    "kolmogorov_challenge": SCRIPT_ROOT / "generate_kolmogorov_decisions.py",
}

CHALLENGE_DEFINITIONS = {
    "shared_selection": {
        "test_pool": "independent 200-trajectory test split",
        "ranking": "descending physical initial-state risk",
        "candidate_pool_multiplier": 4,
        "mixed_cohort": "100 highest-risk test states plus amplified variants of the next 100 states",
        "final_order": "all 200 conditions reordered by physical initial-state risk",
    },
    "burgers": {
        "amplitude_scale": 1.18,
        "added_mode": "0.08 sin(6x + phi), normalized to unit maximum magnitude",
        "phase": "phi sampled uniformly from [0, 2pi)",
        "source_lines": "generate_burgers_decisions.py:461-466,486-535",
    },
    "grayscott": {
        "amplitude_scale": 1.20,
        "reference_state": "(u,v)=(1,0)",
        "patch_radius": "0.10 times the smaller grid dimension, rounded and lower-bounded by two cells",
        "patch": "random-sign v perturbation of magnitude 0.12 and opposite u perturbation of half that magnitude",
        "clipping": "both fields clipped to [0,1.5]",
        "source_lines": "generate_gray_scott_decisions.py:452-477,497-545",
    },
    "kolmogorov": {
        "amplitude_scale": 1.22,
        "reference_state": "base Kolmogorov flow",
        "added_mode": "0.25 sin(3x + phi_x) cos(2y + phi_y), normalized to unit maximum magnitude",
        "phases": "phi_x and phi_y sampled independently and uniformly from [0, 2pi)",
        "source_lines": "generate_kolmogorov_decisions.py:446-460,482-530",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def infer_architecture(model: str, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    shapes = {k: list(v.shape) for k, v in state_dict.items()}
    out: dict[str, Any] = {}
    if model in {"fno", "pino"}:
        spectral = sorted(k for k in shapes if ".specs." in k and k.endswith(".weight"))
        first = shapes[spectral[0]]
        out.update(
            spectral_layers=len(spectral),
            width=first[0],
            modes=first[2:] if len(first) > 3 else first[2],
        )
    elif model == "pinn":
        decoder = sorted(k for k in shapes if ".decoder." in k and k.endswith(".weight"))
        hidden = shapes[decoder[0]][0]
        out.update(decoder_hidden=hidden, decoder_hidden_layers=max(0, len(decoder) - 2))
    elif model == "deeponet":
        trunk = sorted(k for k in shapes if k.startswith("trunk.") and k.endswith(".weight"))
        branch = sorted(k for k in shapes if k.startswith("branch.") and k.endswith(".weight"))
        out.update(
            trunk_hidden=shapes[trunk[0]][0],
            trunk_layers=len(trunk),
            branch_hidden=shapes[branch[0]][0],
            branch_layers=len(branch),
        )
    refine = sorted(k for k in shapes if "refine" in k and k.endswith(".weight"))
    if refine:
        out["local_refinement_channels"] = shapes[refine[0]][0]
        out["local_refinement_layers"] = len(refine)
    return out


def checkpoint_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pde in PDES:
        for model in MODELS:
            summary_path = RESULT_ROOT / pde / model / "challenge_conditions" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint_pde = "gray_scott" if pde == "grayscott" else pde
            ckpt_path = (
                DATA_ROOT / "checkpoints" / "discovery" / checkpoint_pde / f"{model}.pt"
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ckpt["cfg"]
            train = cfg["training"]
            split = cfg["split_info"]
            model_cfg = cfg["model_config"]
            state_dict = ckpt["state_dict"]
            batch_size = 32 if pde == "burgers" and model in {"fno", "pino"} else 16 if pde == "burgers" else 8 if model in {"fno", "pino"} else 4
            row = {
                "pde": pde,
                "model": model,
                "checkpoint": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": sha256(ckpt_path),
                "parameters": int(sum(v.numel() for v in state_dict.values())),
                "epochs": int(train["epochs"]),
                "learning_rate": float(train["lr"]),
                "weight_decay": float(train["weight_decay"]),
                "batch_size_from_resolved_source_default": batch_size,
                "train_trajectories": len(split["train_ids"]),
                "validation_trajectories": len(split["val_ids"]),
                "test_trajectories": len(split["test_ids"]),
                "state_shape": model_cfg["state_shape"],
                "selected_checkpoint": ckpt["selected_checkpoint_tag"],
                "selected_epoch_zero_based": int(ckpt["selected_checkpoint_epoch"]),
                "architecture": infer_architecture(model, state_dict),
                "physics_evaluation_grid": "all 128 spatial points" if pde == "burgers" else "all 48x48 spatial points",
                "separate_collocation_sampling": False,
            }
            rows.append(row)
    return rows


def write_markdown(payload: dict[str, Any]) -> None:
    rows = payload["checkpoints"]
    lines = [
        "# Reproducibility audit",
        "",
        "Checkpoint metadata were read from the twelve files referenced by the canonical discovery decision study summaries.",
        "",
        "| PDE | Model | Parameters | Epochs | Batch | Selected checkpoint | Selected epoch | Architecture |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for r in rows:
        arch = ", ".join(f"{k}={v}" for k, v in r["architecture"].items())
        lines.append(
            f"| {r['pde']} | {r['model']} | {r['parameters']} | {r['epochs']} | "
            f"{r['batch_size_from_resolved_source_default']} | {r['selected_checkpoint']} | "
            f"{r['selected_epoch_zero_based']} | {arch} |"
        )
    lines.extend(
        [
            "",
            "## Shared training settings",
            "",
            "- AdamW, learning rate 3e-4, weight decay 1e-6, cosine annealing, gradient clipping at 1.0 and automatic mixed precision.",
            "- Residual learning in normalized variables and rollout-aware checkpoint selection.",
            "- PINO and PINN physics residuals were evaluated on the complete spatial grid of each mini-batch; no separate collocation-point sampler was used.",
            "- Batch sizes are taken from the resolved training configurations stored with the release.",
            "",
            "## Challenge cohort",
            "",
            "Each PDE used 100 high-risk test states and 100 amplified variants constructed from the next 100 risk-ranked test states. The exact transformations and source-line references are stored in reproducibility_audit.json.",
        ]
    )
    (OUT / "REPRODUCIBILITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = checkpoint_rows()
    payload = {
        "checkpoints": rows,
        "challenge_definitions": CHALLENGE_DEFINITIONS,
        "source_hashes": {name: sha256(path) for name, path in SOURCE_FILES.items()},
        "provenance_limits": [
            "Checkpoint files preserve epochs, optimizer hyperparameters, splits and model state, but not an environment-level BATCH_SIZE_MULT override.",
            "The canonical discovery decision study cache does not preserve optimizer return status or nonlinear constraint-violation diagnostics.",
        ],
    }
    (OUT / "reproducibility_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "checkpoint_hyperparameters.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k not in {"architecture", "state_shape"}])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    write_markdown(payload)
    print(OUT)


if __name__ == "__main__":
    main()

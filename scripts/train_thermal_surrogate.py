#!/usr/bin/env python3
"""Train a controlled FNO on the frozen thermal study thermal trajectory dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import irfft2, rfft2
from torch.utils.data import DataLoader, Dataset


BASE = Path(__file__).resolve().parents[1]


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        batch, _, height, width = x.shape
        with torch.amp.autocast("cuda", enabled=False):
            x_ft = rfft2(x.float(), dim=(-2, -1))
            out_ft = torch.zeros(
                batch,
                self.out_channels,
                height,
                width // 2 + 1,
                device=x.device,
                dtype=torch.cfloat,
            )
            m1 = min(self.modes, x_ft.shape[-2])
            m2 = min(self.modes, x_ft.shape[-1])
            out_ft[:, :, :m1, :m2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weight[:, :, :m1, :m2]
            )
            out = irfft2(out_ft, s=(height, width), dim=(-2, -1))
        return out.to(out_dtype)


class ControlledThermalFNO(nn.Module):
    def __init__(self, width: int = 32, modes: int = 12, layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(5, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.projection = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            x = F.gelu(spectral(x) + local(x))
        return self.projection(x)


class PairDataset(Dataset):
    def __init__(self, state: np.ndarray, control: np.ndarray, trajectory_ids: np.ndarray):
        self.state = state
        self.control = control
        self.index = [
            (int(trajectory), step)
            for trajectory in trajectory_ids
            for step in range(control.shape[1])
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        trajectory, step = self.index[index]
        return (
            torch.from_numpy(self.state[trajectory, step]),
            torch.tensor(self.control[trajectory, step], dtype=torch.float32),
            torch.from_numpy(self.state[trajectory, step + 1]),
            torch.tensor(trajectory, dtype=torch.long),
            torch.tensor(step, dtype=torch.long),
        )


def make_coords(size: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, size, device=device)
    y = torch.linspace(-1.0, 1.0, size, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([X, Y], dim=0)


def normalize_stats(state: np.ndarray, control: np.ndarray, train_ids: np.ndarray) -> dict[str, np.ndarray]:
    train_state = np.asarray(state[train_ids], dtype=np.float64)
    delta = train_state[:, 1:] - train_state[:, :-1]
    train_control = np.asarray(control[train_ids], dtype=np.float64)
    stats = {
        "state_mean": np.mean(train_state, axis=(0, 1, 3, 4)).astype(np.float32),
        "state_std": np.std(train_state, axis=(0, 1, 3, 4)).astype(np.float32),
        "delta_mean": np.mean(delta, axis=(0, 1, 3, 4)).astype(np.float32),
        "delta_std": np.std(delta, axis=(0, 1, 3, 4)).astype(np.float32),
        "control_mean": np.asarray([np.mean(train_control)], dtype=np.float32),
        "control_std": np.asarray([np.std(train_control)], dtype=np.float32),
    }
    for key in ("state_std", "delta_std", "control_std"):
        stats[key] = np.maximum(stats[key], 1e-6)
    return stats


def tensor_stats(stats: dict[str, np.ndarray], device: torch.device):
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in stats.items()}


def predict_next(
    model: nn.Module,
    state: torch.Tensor,
    control: torch.Tensor,
    stats: dict[str, torch.Tensor],
    coords: torch.Tensor,
    project: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = state.shape
    state_n = (state - stats["state_mean"][None, :, None, None]) / stats["state_std"][None, :, None, None]
    control_n = (control.reshape(batch, 1) - stats["control_mean"]) / stats["control_std"]
    control_plane = control_n[:, :, None, None].expand(batch, 1, height, width)
    coord_plane = coords[None].expand(batch, 2, height, width)
    delta_n = model(torch.cat([state_n, control_plane, coord_plane], dim=1))
    delta = delta_n * stats["delta_std"][None, :, None, None] + stats["delta_mean"][None, :, None, None]
    next_state = state + delta
    if project:
        temperature = torch.clamp(next_state[:, 0:1], 0.03, 4.0)
        concentration = torch.clamp(next_state[:, 1:2], 0.0, 1.5)
        next_state = torch.cat([temperature, concentration], dim=1)
    return next_state, delta_n


def rollout_model(
    model: nn.Module,
    initial: torch.Tensor,
    controls: torch.Tensor,
    stats: dict[str, torch.Tensor],
    coords: torch.Tensor,
) -> torch.Tensor:
    current = initial
    states = [current]
    for step in range(controls.shape[1]):
        current, _ = predict_next(model, current, controls[:, step], stats, coords, project=True)
        states.append(current)
    return torch.stack(states, dim=1)


@torch.no_grad()
def trajectory_metrics(
    model: nn.Module,
    state: np.ndarray,
    control: np.ndarray,
    ids: np.ndarray,
    stats_t: dict[str, torch.Tensor],
    coords: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predicted = []
    for start in range(0, len(ids), 8):
        local_ids = ids[start : start + 8]
        initial = torch.from_numpy(state[local_ids, 0]).to(device)
        controls = torch.from_numpy(control[local_ids]).to(device)
        predicted.append(rollout_model(model, initial, controls, stats_t, coords).cpu().numpy())
    pred = np.concatenate(predicted, axis=0)
    true = state[ids]
    rel_l2 = float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))
    pred_risk = np.max(pred[:, :, 0], axis=(1, 2, 3))
    true_risk = np.max(true[:, :, 0], axis=(1, 2, 3))
    risk_mae = float(np.mean(np.abs(pred_risk - true_risk)))
    pred_safe = pred_risk <= 2.0
    true_safe = true_risk <= 2.0
    false_safe = int(np.sum(pred_safe & ~true_safe))
    safe_disagreement = int(np.sum(pred_safe != true_safe))
    return {
        "rollout_rel_l2": rel_l2,
        "risk_mae": risk_mae,
        "false_safe": false_safe,
        "safety_disagreement": safe_disagreement,
        "n_trajectories": len(ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=BASE / "external_data/datasets/thermal/arrhenius_reaction_diffusion_trajectories.npz",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=BASE / "outputs/training/thermal_fno",
    )
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rollout-weight", type=float, default=0.25)
    parser.add_argument("--rollout-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = np.load(args.data, allow_pickle=False)
    state = np.asarray(data["state"], dtype=np.float32)
    control = np.asarray(data["control"], dtype=np.float32)
    n = state.shape[0]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))
    train_ids = np.sort(order[:n_train])
    val_ids = np.sort(order[n_train : n_train + n_val])
    test_ids = np.sort(order[n_train + n_val :])
    stats_np = normalize_stats(state, control, train_ids)
    stats_t = tensor_stats(stats_np, device)
    coords = make_coords(state.shape[-1], device)

    train_dataset = PairDataset(state, control, train_ids)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = ControlledThermalFNO(args.width, args.modes, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Complex spectral weights are incompatible with CUDA GradScaler's foreach
    # unscale kernel. Keep the held-out model in FP32 for deterministic training.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_score = math.inf
    history = []
    args.outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.outdir / "best_thermal_fno.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch_index, (state0, action, state1, trajectory, step) in enumerate(loader):
            state0 = state0.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            state1 = state1.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=False):
                predicted, predicted_delta_n = predict_next(
                    model, state0, action, stats_t, coords, project=False
                )
                true_delta = state1 - state0
                true_delta_n = (
                    true_delta - stats_t["delta_mean"][None, :, None, None]
                ) / stats_t["delta_std"][None, :, None, None]
                one_step = F.mse_loss(predicted_delta_n, true_delta_n)
                risk_loss = F.l1_loss(
                    torch.amax(predicted[:, 0], dim=(-2, -1)),
                    torch.amax(state1[:, 0], dim=(-2, -1)),
                )
                loss = one_step + 0.20 * risk_loss

                if args.rollout_weight > 0 and batch_index % 4 == 0:
                    valid = (step.numpy() + args.rollout_k) <= control.shape[1]
                    ids = np.flatnonzero(valid)[:8]
                    if len(ids):
                        tr = trajectory.numpy()[ids]
                        tt = step.numpy()[ids]
                        current = state0[ids]
                        rollout_losses = []
                        for offset in range(args.rollout_k):
                            action_np = control[tr, tt + offset]
                            target_np = state[tr, tt + offset + 1]
                            action_t = torch.from_numpy(action_np).to(device)
                            target_t = torch.from_numpy(target_np).to(device)
                            current, _ = predict_next(
                                model, current, action_t, stats_t, coords, project=True
                            )
                            rollout_losses.append(
                                F.mse_loss(
                                    (current - stats_t["state_mean"][None, :, None, None])
                                    / stats_t["state_std"][None, :, None, None],
                                    (target_t - stats_t["state_mean"][None, :, None, None])
                                    / stats_t["state_std"][None, :, None, None],
                                )
                            )
                        loss = loss + args.rollout_weight * torch.stack(rollout_losses).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            metrics = trajectory_metrics(model, state, control, val_ids, stats_t, coords, device)
            score = metrics["rollout_rel_l2"] + 2.0 * metrics["risk_mae"]
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "selection_score": score,
                **metrics,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if score < best_score:
                best_score = score
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_args": {"width": args.width, "modes": args.modes, "layers": args.layers},
                        "stats": {key: value.tolist() for key, value in stats_np.items()},
                        "split": {
                            "train_ids": train_ids.tolist(),
                            "val_ids": val_ids.tolist(),
                            "test_ids": test_ids.tolist(),
                        },
                        "data": str(args.data),
                        "epoch": epoch,
                        "selection_score": score,
                        "seed": args.seed,
                    },
                    checkpoint_path,
                )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = trajectory_metrics(model, state, control, test_ids, stats_t, coords, device)
    summary = {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint["epoch"],
        "best_validation_selection_score": checkpoint["selection_score"],
        "split_sizes": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "test_metrics": test_metrics,
        "history": history,
    }
    (args.outdir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

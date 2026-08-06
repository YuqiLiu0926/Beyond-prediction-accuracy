"""Residual-feature mismatch detection used in the recoverability study.

The public detector combines surrogate risk, surrogate safety margin, nineteen
operator-defect features, and one-hot indicators for the PDE and surrogate
architecture. Calibration is performed on physical-state groups so that the
four surrogate evaluations sharing one state remain dependent observations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


PDES = ("burgers", "grayscott", "kolmogorov")
MODELS = ("deeponet", "fno", "pinn", "pino")


def _one_hot(value: str, names: tuple[str, ...]) -> list[float]:
    return [1.0 if value == name else 0.0 for name in names]


def clean_features(values: np.ndarray) -> np.ndarray:
    """Replace nonfinite feature values using fixed numeric sentinels."""
    return np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=0.0,
        posinf=1e8,
        neginf=-1e8,
    )


def residual_feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    """Construct the frozen detector input matrix from archived replay rows."""
    features = []
    for row in rows:
        residual = [float(value) for value in row["residual_features"]]
        if len(residual) != 19:
            raise ValueError(
                f"Expected 19 operator-defect features, found {len(residual)}"
            )
        features.append(
            [float(row["m_sur_replay"]), float(row["z_sur_replay"])]
            + residual
            + _one_hot(str(row["pde"]), PDES)
            + _one_hot(str(row["model"]), MODELS)
        )
    return clean_features(np.asarray(features, dtype=np.float64))


def fit_gap_model(
    features: np.ndarray,
    target_gap: np.ndarray,
    seed: int = 42,
) -> GradientBoostingRegressor:
    """Fit the frozen Huber gradient-boosting risk-gap regressor."""
    model = GradientBoostingRegressor(
        loss="huber",
        alpha=0.90,
        n_estimators=120,
        max_depth=2,
        learning_rate=0.035,
        min_samples_leaf=8,
        subsample=0.85,
        random_state=seed,
    )
    model.fit(clean_features(features), np.asarray(target_gap, dtype=np.float64))
    return model


def finite_sample_upper_quantile(
    scores: np.ndarray,
    coverage: float,
) -> tuple[float, int, int, bool]:
    """Return the split-conformal upper quantile with finite-sample rank."""
    values = np.asarray(scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("inf"), 1, 0, False
    rank = int(math.ceil((n + 1) * float(coverage)))
    if rank > n:
        return float("inf"), rank, n, False
    rank = max(rank, 1)
    return float(np.partition(values, rank - 1)[rank - 1]), rank, n, True

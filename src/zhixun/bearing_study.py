"""Local B RUL diagnostics and device-weighted continuation from the official model."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from zhixun.data import Dataset, load_dataset
from zhixun.vendor.data import json_value
from zhixun.vendor.model import Model


def load_bearings(path: Path, model: Model) -> tuple[Dataset, np.ndarray]:
    data = load_dataset(path, model, "B")
    identifiers, bearings = [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json_value(line)
            bearing = row["meta"].get("bearing")
            if not isinstance(bearing, str) or not bearing.strip():
                raise ValueError("B study requires the official bearing metadata")
            identifiers.append(row["id"])
            bearings.append(bearing)
    if tuple(identifiers) != data.ids or len(set(bearings)) != len(data.groups):
        raise ValueError("Bearing metadata does not match the official rows and groups")
    return data, np.asarray(bearings)


def bearing_weights(bearings: np.ndarray, fraction: float) -> np.ndarray:
    """Mix sample-mean MSE and the mean of per-bearing MSEs, with mean weight one."""
    if bearings.ndim != 1 or not len(bearings) or not 0 <= fraction <= 1:
        raise ValueError("Invalid bearing groups or weighting fraction")
    _, inverse, counts = np.unique(bearings, return_inverse=True, return_counts=True)
    return 1 - fraction + fraction * (len(bearings) / (len(counts) * counts[inverse]))


def fit_weighted_bearing_epoch(model: Model, data: Dataset, weights: np.ndarray) -> float:
    """Resume one epoch, preserving vendor shuffling, clipping and Adam state."""
    if (
        model.meta.get("profile") != "B" or model.meta["task"] != "regression"
        or not len(data.y) or data.y.shape != (len(data.x),)
        or weights.shape != data.y.shape or not np.isfinite(weights).all()
        or np.any(weights <= 0) or not np.isclose(weights.mean(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Expected a B checkpoint and aligned positive, mean-one weights")
    x = model.normalized(data.x)
    y = (data.y - model.meta["target_mean"]) / model.meta["target_scale"]
    rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
    order = rng.permutation(len(y))
    loss_sum = 0.0
    batch_size = model.meta["batch_size"]
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch = x[indices]
        hidden, logits = model.forward(batch)
        error = logits[:, 0] - y[indices]
        batch_weights = weights[indices]
        loss_sum += float(np.sum(error ** 2 * batch_weights))
        # Normalize by batch size, not the varying sum of the batch's weights.
        delta = (2 * error * batch_weights / len(indices))[:, None]
        hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
        gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0),
                     "w1": batch.T @ hidden_delta, "b1": hidden_delta.sum(axis=0)}
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
        if not math.isfinite(norm):
            raise ValueError("Nonfinite weighted gradient")
        model.meta["steps"] += 1
        for name, gradient in gradients.items():
            gradient *= min(1.0, 5.0 / max(norm, 1e-12))
            model.first[name] = .9 * model.first[name] + .1 * gradient
            model.second[name] = .999 * model.second[name] + .001 * gradient ** 2
            first = model.first[name] / (1 - .9 ** model.meta["steps"])
            second = model.second[name] / (1 - .999 ** model.meta["steps"])
            model.params[name] -= model.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
        if not all(np.isfinite(p).all() for p in model.params.values()):
            raise ValueError("Nonfinite weighted model state")
    model.meta["epochs"] += 1
    model.validate()
    return loss_sum / len(y)


def _errors(error: np.ndarray) -> dict:
    over = error[error > 0]
    return {
        "rows": len(error),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "overestimate_fraction": float(np.mean(error > 0)),
        "overestimate_p95": float(np.quantile(over, .95)) if len(over) else None,
    }


def bearing_metrics(prediction: np.ndarray, target: np.ndarray, bearings: np.ndarray) -> dict:
    if (prediction.shape != target.shape or target.shape != bearings.shape or target.ndim != 1
            or not len(target) or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError("Predictions, targets and groups must be finite aligned vectors")
    error = prediction - target
    by_bearing = {str(g): _errors(error[bearings == g]) for g in np.unique(bearings)}
    ranges = {}
    for low, high in ((0, 100), (100, 500), (500, 1000), (1000, math.inf)):
        selected = (target >= low) & (target < high)
        if selected.any():
            ranges[f"{low}-{'inf' if math.isinf(high) else high}"] = _errors(error[selected])
    return {
        "overall": _errors(error), "bearings": by_bearing, "rul_ranges_minutes": ranges,
        "bearing_mean_rmse": float(np.mean([m["rmse"] for m in by_bearing.values()])),
        "bearing_worst_rmse": max(m["rmse"] for m in by_bearing.values()),
    }


def bearing_gate(before: dict, after: dict) -> dict:
    """Development screen on the same public panel, with safeguards against one-device wins."""
    if {g: d["rows"] for g, d in before["bearings"].items()} != {
        g: d["rows"] for g, d in after["bearings"].items()
    }:
        raise ValueError("Group gate requires identical evaluation support")
    eps = 1e-9
    improved = [g for g in before["bearings"]
                if after["bearings"][g]["rmse"] < before["bearings"][g]["rmse"] - eps]
    checks = {
        "overall_rmse_improved": after["overall"]["rmse"] < before["overall"]["rmse"] - eps,
        "overall_mae_not_worse": after["overall"]["mae"] <= before["overall"]["mae"] + eps,
        "two_bearings_improved": len(improved) >= 2,
        "bearing_mean_rmse_improved": after["bearing_mean_rmse"] < before["bearing_mean_rmse"] - eps,
        "worst_bearing_rmse_not_worse": after["bearing_worst_rmse"] <= before["bearing_worst_rmse"] + eps,
    }
    return {"passes": all(checks.values()), "checks": checks, "bearings_improved": improved}

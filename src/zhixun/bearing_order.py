"""Continue the official B network with an optional within-bearing order loss."""

from __future__ import annotations

import math

import numpy as np

from zhixun.data import Dataset
from zhixun.vendor.model import Model


def pairing_pools(target: np.ndarray, bearings: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use only observed training labels to pair early and late life in the same device."""
    if target.ndim != 1 or target.shape != bearings.shape or not np.isfinite(target).all():
        raise ValueError("Aligned finite rank labels are required")
    pools = []
    for name in np.unique(bearings):
        rows = np.flatnonzero(bearings == name)
        lo, hi = np.quantile(target[rows], [.25, .75])
        if hi > lo:
            pools.append((rows[target[rows] >= hi], rows[target[rows] <= lo]))
    if len(pools) < 2:
        raise ValueError("Ranking needs distinct lifetime stages from at least two training devices")
    return pools


def fit_ordered(
    model: Model, data: Dataset, weights: np.ndarray, pools, *, epochs: int, penalty: float,
) -> list[dict]:
    """Match the reference optimizer and add a group-balanced margin loss on original rows."""
    y = (data.y - model.meta["target_mean"]) / model.meta["target_scale"]
    if (model.meta.get("profile") != "B" or data.x.shape[0] != len(y)
            or weights.shape != y.shape or np.any(weights <= 0)
            or not np.isclose(weights.mean(), 1) or not math.isfinite(penalty) or penalty < 0
            or type(epochs) is not int or not 1 <= epochs <= 500):
        raise ValueError("Invalid ordered training settings")
    x = model.normalized(data.x)
    history = []
    for _ in range(epochs):
        seed, epoch = model.meta["seed"], model.meta["epochs"]
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(y))
        pairing = np.random.default_rng(np.random.SeedSequence([seed, epoch, 11]))
        mse_sum, violation_sum, steps = 0., 0., 0
        batch_size = model.meta["batch_size"]
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            n = len(rows)
            pairs = 0
            if penalty:
                # Four pairs per bearing per optimizer step, independent of trajectory length.
                high = np.concatenate([pairing.choice(h, 4) for h, _ in pools])
                low = np.concatenate([pairing.choice(l, 4) for _, l in pools])
                pairs = len(high)
                indices = np.concatenate([rows, high, low])
            else:
                indices = rows
            batch = x[indices]
            hidden, raw = model.forward(batch)
            prediction = raw[:, 0]
            error = prediction[:n] - y[rows]
            mse_sum += float(np.sum(error ** 2 * weights[rows]))
            delta = np.empty(len(indices), dtype=np.float64)
            delta[:n] = 2 * error * weights[rows] / n
            if pairs:
                # A minimum positive normalized prediction gap for well-separated stages.
                violation = np.maximum(0., .25 - prediction[n:n + pairs] + prediction[n + pairs:])
                delta[n:n + pairs] = -2 * penalty * violation / pairs
                delta[n + pairs:] = -delta[n:n + pairs]
                violation_sum += float(np.mean(violation ** 2))
            delta = delta[:, None]
            hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0),
                         "w1": batch.T @ hidden_delta, "b1": hidden_delta.sum(axis=0)}
            norm = math.sqrt(sum(float(np.sum(v * v)) for v in gradients.values()))
            if not math.isfinite(norm):
                raise ValueError("Ordered training produced a nonfinite gradient")
            model.meta["steps"] += 1
            steps += 1
            for name, grad in gradients.items():
                grad *= min(1.0, 5.0 / max(norm, 1e-12))
                model.first[name] = .9 * model.first[name] + .1 * grad
                model.second[name] = .999 * model.second[name] + .001 * grad ** 2
                first = model.first[name] / (1 - .9 ** model.meta["steps"])
                second = model.second[name] / (1 - .999 ** model.meta["steps"])
                model.params[name] -= model.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
        model.meta["epochs"] += 1
        history.append({"weighted_mse": mse_sum / len(y), "order_violation_mse": violation_sum / steps})
    model.validate()
    return history

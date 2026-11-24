"""Test a no-op-at-migration residual MLP basis with a standard B checkpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time
from uuid import uuid4

from run_b_residual import PREVIOUS, ROOT, record, show

import numpy as np

from zhixun.b_residual import append_residual_basis, fit_residual_outputs
from zhixun.bearing_study import bearing_gate, bearing_metrics, bearing_weights, fit_weighted_bearing_epoch, load_bearings
from zhixun.data import digest, verify_inputs, verify_parent, verify_split, write_json
from zhixun.vendor.model import Model


def main():
    verify_inputs("B", ROOT)
    train_file, val_file = ROOT / "data/B/train.jsonl", ROOT / "data/B/validation.jsonl"
    parent = Model.load(PREVIOUS)
    train, tg = load_bearings(train_file, parent)
    val, vg = load_bearings(val_file, parent)
    verify_parent(parent, "B", train)
    verify_split(train, val)
    out = ROOT / "artifacts" / f"b-residual-basis-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    registration = {
        "study": out.name, "strategy": "train_quantile_tanh_residual_basis",
        "penalty": .03, "training_weight_bearing_fraction": .5, "target_unit": "minutes", "threads": 1,
        "continuation_epoch": 5, "continuation_lr": .0001, "parent_sha256": digest(PREVIOUS),
        "train_sha256": digest(train_file), "validation_sha256": digest(val_file),
        "source_sha256": digest(ROOT/"src/zhixun/b_residual.py"), "evaluation": "official_public_B_exploratory",
    }
    write_json(out / "registration.json", registration)
    print("STUDY", out, flush=True)
    start_cpu, start_wall = process_time(), perf_counter()
    model = append_residual_basis(parent, train.x)
    expected = parent.predict(val.x)
    migration_delta = float(np.max(np.abs(model.predict(val.x) - expected)))
    assert migration_delta < 1e-9
    assert model.meta["steps"] == parent.meta["steps"] and model.meta["epochs"] == parent.meta["epochs"]
    weight = bearing_weights(tg, .5)
    fit_residual_outputs(model, train.x, train.y, weight, parent_width=parent.meta["hidden"], penalty=.03)
    parent_metrics = bearing_metrics(expected, val.y, vg)
    runs = []
    for stage in ("ridge-only", "ridge-adam"):
        losses = []
        if stage == "ridge-adam":
            model.meta["learning_rate"] = .0001
            losses = [fit_weighted_bearing_epoch(model, train, weight) for _ in range(5)]
        cpu, wall = process_time() - start_cpu, perf_counter() - start_wall
        metrics = bearing_metrics(model.predict(val.x), val.y, vg)
        name = f"basis-{stage}"
        folder = out / name
        folder.mkdir()
        checkpoint = folder / "checkpoint.npz"
        model.save(checkpoint)
        assert np.allclose(Model.load(checkpoint).predict(val.x), model.predict(val.x), rtol=0, atol=1e-9)
        result = {"name": name, "role": "explicit_residual_continuation_development", "validation": metrics,
                  "training_cpu_seconds": cpu, "training_wall_seconds": wall,
                  "losses": losses, "configuration": {"hidden": model.meta["hidden"], "adam_epochs": len(losses),
                                                        "migration_max_difference": migration_delta},
                  "gate_vs_B2": bearing_gate(parent_metrics, metrics), "checkpoint_sha256": digest(checkpoint)}
        runs.append(result)
        show(name, metrics)
        start_cpu, start_wall = process_time(), perf_counter()
    summary = {"registration": registration, "B2": parent_metrics, "runs": runs}
    write_json(out / "summary.json", summary)
    for result in runs:
        folder = out / result["name"]
        result["swanlab"] = record(folder, result, registration)
        write_json(folder / "result.json", result)
        write_json(out / "summary.json", summary)
    print("RESULTS", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()

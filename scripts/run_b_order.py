"""Compare duration-matched B continuation with and without within-bearing ordering."""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["BASELINE_CPU_THREADS"] = "1"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from zhixun.bearing_order import fit_ordered, pairing_pools
from zhixun.bearing_study import bearing_gate, bearing_metrics, bearing_weights, load_bearings
from zhixun.data import digest, verify_inputs, verify_parent, verify_split, write_json
from zhixun.model_extension import expand_hidden
from zhixun.vendor.model import Model


def evaluate(model, validation, groups):
    p = model.predict(validation.x)
    report = bearing_metrics(p, validation.y, groups)
    report["per_bearing_rul_correlation"] = {}
    for name in np.unique(groups):
        selected = groups == name
        predicted, true = p[selected], validation.y[selected]
        report["per_bearing_rul_correlation"][str(name)] = (
            float(np.corrcoef(predicted, true)[0, 1]) if predicted.std() and true.std() else None
        )
    return report


def main():
    base_path = ROOT / "data/B/checkpoint.npz"
    b2_path = ROOT / "artifacts/b-phase1-20261123T142953-3f0ddb42/B2-balanced/checkpoint.npz"
    train_path, val_path = ROOT / "data/B/train.jsonl", ROOT / "data/B/validation.jsonl"
    source = verify_inputs("B", ROOT)
    base = Model.load(base_path)
    train, train_groups = load_bearings(train_path, base)
    validation, validation_groups = load_bearings(val_path, base)
    verify_parent(base, "B", train)
    verify_split(train, validation)
    b2 = evaluate(Model.load(b2_path), validation, validation_groups)
    weights = bearing_weights(train_groups, .5)
    pools = pairing_pools(train.y, train_groups)
    names = (
        ("B-order-24-control", 24, 0.), ("B-order-24-pair", 24, 1.),
        ("B-order-96-control", 96, 0.), ("B-order-96-pair", 96, 1.),
    )
    out = ROOT / "artifacts" / f"b-order-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "study": out.name, "source_commit": "271263719f06c6b684a30c4d5421035b06e93369",
        "source": source, "parent_sha256": digest(base_path), "b2_sha256": digest(b2_path),
        "train_sha256": digest(train_path), "validation_sha256": digest(val_path),
        "epochs": 90, "learning_rate": .002, "device_mean_loss_fraction": .5,
        "scope": "official_public_validation_exploratory", "plans": names,
    }
    write_json(out / "registration.json", config)
    print(out, flush=True)
    results = []
    for name, width, penalty in names:
        folder = out / name
        folder.mkdir()
        model = expand_hidden(base, width) if width != base.meta["hidden"] else deepcopy(base)
        initial = model.meta["epochs"], model.meta["steps"]
        init_delta = float(np.max(np.abs(model.predict(validation.x) - base.predict(validation.x))))
        if init_delta > 1e-9:
            raise ValueError("Migration did not preserve the starting predictions")
        model.meta["learning_rate"] = config["learning_rate"]
        cpu, wall = process_time(), perf_counter()
        curves = fit_ordered(model, train, weights, pools, epochs=config["epochs"], penalty=penalty)
        cpu, wall = process_time() - cpu, perf_counter() - wall
        after = evaluate(model, validation, validation_groups)
        model.save(folder / "checkpoint.npz")
        rec = {
            "run": name, "hidden": width, "order_penalty": penalty,
            "learning_rate": config["learning_rate"], "parent_sha256": digest(base_path),
            "completed_epochs": model.meta["epochs"] - initial[0],
            "optimizer_steps_added": model.meta["steps"] - initial[1],
            "training_cpu_seconds": cpu, "training_wall_seconds": wall,
            "migration_delta": init_delta, "after": after,
            "gate_vs_B2": bearing_gate(b2, after),
            "checkpoint_sha256": digest(folder / "checkpoint.npz"),
        }
        with (folder / "history.jsonl").open("x", encoding="utf-8") as log:
            for step, curve in enumerate(curves, 1):
                log.write(json.dumps({"step": step, "metrics": {
                    f"training/{key}": value for key, value in curve.items()
                }}, allow_nan=False) + "\n")
        write_json(folder / "result.json", rec)
        results.append(rec)
        write_json(out / "summary.json", {"registration": config, "b2": b2, "runs": results})
        print(name, round(after["overall"]["rmse"], 3), round(after["bearing_mean_rmse"], 3),
              rec["gate_vs_B2"]["passes"], flush=True)


if __name__ == "__main__":
    main()

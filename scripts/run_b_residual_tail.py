"""Compare two fixed asymmetric B residual continuations against B2."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time
from types import SimpleNamespace
from uuid import uuid4

from run_b_residual import PREVIOUS, PROJECT, ROOT, WORKSPACE, show

import numpy as np

from zhixun.b_residual import append_residual_basis
from zhixun.b_residual_tail import fit_asymmetric_residual
from zhixun.bearing_study import bearing_gate, bearing_metrics, bearing_weights, load_bearings
from zhixun.contest_runtime import predict
from zhixun.data import digest, verify_inputs, verify_parent, verify_split, write_json
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model


ARMS = (("over2", 2), ("over4", 4))
PENALTY = 0.03


def record_tail_run(path, result, registration):
    config = {**registration, **result["configuration"], "run": path.name}
    recorder = Recorder(path, config, mode="offline", project=PROJECT, workspace=WORKSPACE)
    try:
        for step, loss in enumerate(result["losses"], 1):
            recorder.log(step, {"training/objective": loss})
        final = result["validation"]
        tail = final["rul_ranges_minutes"]["0-100"]
        metrics = {f"validation/{key}": value for key, value in final["overall"].items() if value is not None}
        metrics.update({f"validation/near_failure_{key}": value for key, value in tail.items() if value is not None})
        metrics.update({f"validation/bearing/{name}/rmse": bearing["rmse"] for name, bearing in final["bearings"].items()})
        metrics.update({"validation/bearing_mean_rmse": final["bearing_mean_rmse"],
                        "validation/bearing_worst_rmse": final["bearing_worst_rmse"],
                        "training/cpu_seconds": result["training_cpu_seconds"],
                        "training/wall_seconds": result["training_wall_seconds"]})
        recorder.log(len(result["losses"]) + 1, metrics)
    except Exception:
        recorder.finish(failed=True)
        raise
    recorder.finish()
    return {**recorder.reference(), "configuration": config, "directory": str(path / "swanlog")}


def main():
    verify_inputs("B", ROOT)
    train_file, val_file = ROOT / "data/B/train.jsonl", ROOT / "data/B/validation.jsonl"
    parent = Model.load(PREVIOUS)
    train, train_groups = load_bearings(train_file, parent)
    verify_parent(parent, "B", train)
    weights = bearing_weights(train_groups, .5)
    out = ROOT / "artifacts" / f"b-residual-tail-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    registration = {
        "study": out.name, "strategy": "asymmetric_quantile_tanh_residual", "arms": dict(ARMS),
        "penalty": PENALTY, "training_weight_bearing_fraction": .5, "target_unit": "minutes", "threads": 1,
        "near_failure_train_rul_interval": [0, 100], "solver": "convex_active_set_newton_backtracking",
        "parent_sha256": digest(PREVIOUS), "train_sha256": digest(train_file), "validation_sha256": digest(val_file),
        "basis_sha256": digest(ROOT / "src/zhixun/b_residual.py"),
        "solver_sha256": digest(ROOT / "src/zhixun/b_residual_tail.py"), "script_sha256": digest(Path(__file__)),
        "evaluation": "official_public_B_exploratory",
    }
    write_json(out / "registration.json", registration)
    print("STUDY", out, flush=True)

    runs = []
    for name, multiplier in ARMS:
        start_cpu, start_wall = process_time(), perf_counter()
        model = append_residual_basis(parent, train.x)
        migration_delta = float(np.max(np.abs(model.predict(train.x) - parent.predict(train.x))))
        if migration_delta >= 1e-9:
            raise ValueError("Initial migration changed a parent prediction")
        losses = fit_asymmetric_residual(model, train.x, train.y, weights, parent_width=parent.meta["hidden"],
                                        penalty=PENALTY, multiplier=multiplier)
        cpu, wall = process_time() - start_cpu, perf_counter() - start_wall
        folder = out / name
        folder.mkdir()
        checkpoint = folder / "checkpoint.npz"
        model.save(checkpoint)
        runs.append({"name": name, "role": "explicit_residual_continuation_development", "losses": losses,
                     "training_cpu_seconds": cpu, "training_wall_seconds": wall,
                     "configuration": {"hidden": model.meta["hidden"], "near_failure_positive_error_weight": multiplier,
                                       "migration_max_difference": migration_delta}, "checkpoint_sha256": digest(checkpoint)})

    val, val_groups = load_bearings(val_file, parent)
    verify_split(train, val)
    base = bearing_metrics(parent.predict(val.x), val.y, val_groups)
    base_tail = base["rul_ranges_minutes"]["0-100"]
    samples = [SimpleNamespace(task="regression", inputs=json.loads(line)["inputs"])
               for line in val_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {"registration": registration, "B2": base, "runs": runs}
    for result in runs:
        folder = out / result["name"]
        model = Model.load(folder / "checkpoint.npz")
        prediction = np.asarray([predict(sample, folder)["value"] for sample in samples])
        endpoint_delta = float(np.max(np.abs(prediction - model.predict(val.x))))
        if endpoint_delta >= 1e-9:
            raise ValueError("Batch and frozen per-row predictions differ")
        metrics = bearing_metrics(prediction, val.y, val_groups)
        tail = metrics["rul_ranges_minutes"]["0-100"]
        result.update({"validation": metrics, "gate_vs_B2": bearing_gate(base, metrics),
                       "tail_gate_vs_B2": (tail["rmse"] <= base_tail["rmse"] and
                                           (tail["overestimate_p95"] or 0) <= (base_tail["overestimate_p95"] or 0)),
                       "endpoint_max_difference": endpoint_delta})
        show(result["name"], metrics)
        print(f"  tail RMSE {tail['rmse']:.4f}, overestimate P95 {tail['overestimate_p95']:.4f}", flush=True)
    write_json(out / "summary.json", summary)
    for result in runs:
        folder = out / result["name"]
        result["swanlab"] = record_tail_run(folder, result, registration)
        write_json(folder / "result.json", result)
        write_json(out / "summary.json", summary)
    print("RESULTS", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()

"""Run the B development study, then publish and verify each SwanLab run."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import platform
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for variable in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from zhixun.bearing_study import (
    bearing_gate,
    bearing_metrics,
    bearing_weights,
    fit_weighted_bearing_epoch,
    load_bearings,
)
from zhixun.bearing_swanlab import PROJECT, PROJECT_PATH, WORKSPACE, connect, publish_run
from zhixun.data import digest, verify_inputs, verify_parent, verify_split, write_json
from zhixun.model_extension import expand_hidden
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model


BASE_COMMIT = "271263719f06c6b684a30c4d5421035b06e93369"


def historical_model(reference, inputs):
    root = reference / "references/b-history"
    for name, expected in inputs["historical_artifacts"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Historical input changed: {name}")
    report = json.loads((root / "b-final/training.json").read_text(encoding="utf-8"))
    source = root / "baselines/model.py"
    spec = importlib.util.spec_from_file_location("b_historical_vendor", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent_file = root / "b-expanded96/checkpoint.npz"
    return module.Model.load(parent_file), parent_file, report


def run_arm(root, name, parent, train, validation, groups, *, epochs, rate=None,
            weights=None, cpu_budget=None):
    model = copy.deepcopy(parent)
    before = bearing_metrics(parent.predict(validation.x), validation.y, groups)
    if rate is not None:
        model.meta["learning_rate"] = rate
    epoch_losses = []
    cpu_start, wall_start = process_time(), perf_counter()
    for _ in range(epochs):
        if cpu_budget is not None and epoch_losses and process_time() - cpu_start >= cpu_budget:
            break
        if weights is None:
            loss = model.fit(train.x, train.y, 1)[0]
        else:
            loss = fit_weighted_bearing_epoch(model, train, weights)
        epoch_losses.append(float(loss))
    actual_epochs = len(epoch_losses)
    cpu = process_time() - cpu_start if actual_epochs else 0.0
    seconds = perf_counter() - wall_start if actual_epochs else 0.0
    prediction = model.predict(validation.x)
    expected_steps = actual_epochs * math.ceil(len(train.ids) / parent.meta["batch_size"])
    if (model.meta["steps"] - parent.meta["steps"] != expected_steps
            or model.meta["epochs"] - parent.meta["epochs"] != actual_epochs
            or not np.array_equal(model.mean, parent.mean)
            or not np.array_equal(model.scale, parent.scale)
            or any(model.meta[k] != parent.meta[k] for k in ("target_mean", "target_scale", "seed"))):
        raise ValueError(f"Continuation invariants failed for {name}")
    folder = root / name
    folder.mkdir(exist_ok=False)
    artifact = folder / "checkpoint.npz"
    model.save(artifact)
    restored = Model.load(artifact)
    if not np.allclose(restored.predict(validation.x), prediction, rtol=0, atol=1e-10):
        raise ValueError(f"Reloaded predictions changed for {name}")
    result = {
        "run": name, "hidden": model.meta["hidden"], "parameters": model.meta["parameter_count"],
        "batch_size": model.meta["batch_size"], "seed": model.meta["seed"], "learning_rate": rate,
        "max_requested_epochs": epochs, "completed_epochs": actual_epochs,
        "optimizer_steps_added": expected_steps, "training_cpu_seconds": cpu,
        "training_wall_seconds": seconds, "cpu_budget_seconds": cpu_budget,
        "equal_bearing_loss_fraction": 0.0 if weights is None else 0.5,
        "before": before, "after": bearing_metrics(prediction, validation.y, groups),
        "losses": epoch_losses, "checkpoint_sha256": digest(artifact),
    }
    write_json(folder / "result.json", result)
    return result


def record_swanlab(root, run, common):
    folder = root / run["run"]
    config = {**common, **{k: run[k] for k in (
        "run", "hidden", "parameters", "seed", "batch_size", "learning_rate",
        "completed_epochs", "equal_bearing_loss_fraction", "parent_sha256", "checkpoint_sha256",
    )}}
    recorder = Recorder(folder, config, mode="offline", project=PROJECT, workspace=WORKSPACE)
    try:
        recorder.log(0, {f"before/{k}": float(v) for k, v in run["before"]["overall"].items()
                         if v is not None})
        for step, loss in enumerate(run["losses"], 1):
            recorder.log(step, {"training/objective": loss})
        after = run["after"]
        scalars = {f"validation/{k}": float(v) for k, v in after["overall"].items()
                   if v is not None}
        for name, metric in after["bearings"].items():
            scalars[f"bearing/{name}/rmse"] = metric["rmse"]
            scalars[f"bearing/{name}/rows"] = float(metric["rows"])
        scalars.update({"validation/bearing_mean_rmse": after["bearing_mean_rmse"],
                        "validation/bearing_worst_rmse": after["bearing_worst_rmse"],
                        "training/cpu_seconds": run["training_cpu_seconds"],
                        "training/wall_seconds": run["training_wall_seconds"],
                        "training/actual_epochs": float(run["completed_epochs"])})
        if "gate_vs_B1" in run:
            scalars["decision/passed_development_gate"] = float(run["gate_vs_B1"]["passes"])
        recorder.log(len(run["losses"]) + 1, scalars)
        recorder.text(len(run["losses"]) + 1, "decision/evidence", {
            "scope": "official_public_validation",
            "gate_vs_B1": run.get("gate_vs_B1"),
            "gate_vs_B0": run.get("gate_vs_B0"),
            "historical_reproduction": run.get("historical_reproduction"),
        })
    except Exception:
        recorder.finish(failed=True)
        raise
    recorder.finish()
    return {**recorder.reference(), "directory": str(folder / "swanlog"),
            "configuration": config, "publication_status": "local_ready"}


def publish_study(out, summary):
    pending = [run for run in summary["runs"]
               if not run.get("swanlab", {}).get("cloud_record_verified")]
    if not pending:
        return 0
    for run in pending:
        name = run["run"]
        if not re.fullmatch(r"B[0-3]-[A-Za-z0-9-]+", name) or (out / name).is_symlink():
            raise ValueError("Invalid local arm directory in the study")
    try:
        api = connect()
    except Exception as error:
        # A login or network error affects every arm, so avoid retrying it for each one.
        api, connection_failure = None, type(error).__name__
    failed = 0
    for run in pending:
        tracking = run.get("swanlab")
        if not tracking or not tracking.get("configuration"):
            status = {"status": "local_record_incomplete", "cloud_record_verified": False}
        elif api is None:
            status = {"status": "connection_failed", "failure": connection_failure,
                      "cloud_record_verified": False}
        else:
            print(f"SWANLAB SYNC: {run['run']} -> {PROJECT_PATH}", flush=True)
            try:
                status = publish_run(ROOT, out / run["run"], tracking["configuration"], api)
            except Exception as error:
                status = {"status": "local_record_invalid", "failure": type(error).__name__,
                          "cloud_record_verified": False}
        run["swanlab"] = {
            **(tracking or {}), "publication": status,
            "publication_status": status["status"],
            "cloud_record_verified": status["cloud_record_verified"],
            "cloud_url": status.get("cloud_url"),
        }
        write_json(out / run["run"] / "result.json", run)
        write_json(out / "summary.json", summary)
        failed += not status["cloud_record_verified"]
        print(f"SWANLAB {run['run']}: {status['status']} {status.get('cloud_url', '')}", flush=True)
    return failed


def write_metric_tables(root, runs):
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "hidden", "lr", "actual_epochs", "rmse_min", "mae_min",
                         "bearing_mean_rmse_min", "bearing_worst_rmse_min", "cpu_s", "wall_s",
                         "gate_vs_B1", "checkpoint_sha256"])
        for run in runs:
            a = run["after"]
            writer.writerow([run["run"], run["hidden"], run["learning_rate"], run["completed_epochs"],
                             a["overall"]["rmse"], a["overall"]["mae"], a["bearing_mean_rmse"],
                             a["bearing_worst_rmse"], run["training_cpu_seconds"],
                             run["training_wall_seconds"], run.get("gate_vs_B1", {}).get("passes"),
                             run["checkpoint_sha256"]])
    with (root / "groups_and_life_ranges.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        fields = ["rows", "rmse", "mae", "bias", "overestimate_fraction", "overestimate_p95"]
        writer.writerow(["run", "type", "group", *fields])
        for run in runs:
            for kind in ("bearings", "rul_ranges_minutes"):
                for group, metric in run["after"][kind].items():
                    writer.writerow([run["run"], kind, group, *(metric[f] for f in fields)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Explicitly keep only local journals")
    mode.add_argument("--publish-only", type=Path, metavar="STUDY_DIR",
                      help="Retry pending uploads for a study from this script without retraining")
    args = parser.parse_args()
    if args.publish_only is not None:
        out = args.publish_only.resolve()
        if out.parent != (ROOT / "artifacts").resolve() or not out.name.startswith("b-phase1-"):
            parser.error("Specify a B phase 1 study directory from this project's artifacts")
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        if summary["registration"].get("swanlab_destination") != PROJECT_PATH:
            parser.error("This study predates the automatic publisher or belongs to another project")
        return int(publish_study(out, summary) != 0)

    inputs = json.loads((ROOT / "phase1_inputs.json").read_text(encoding="utf-8"))
    if inputs["source_commit"] != BASE_COMMIT:
        raise ValueError("Unexpected school version")
    declared = verify_inputs("B", ROOT)
    base_file = ROOT / "data/B/checkpoint.npz"
    train_file, val_file = ROOT / "data/B/train.jsonl", ROOT / "data/B/validation.jsonl"
    base = Model.load(base_file)
    train, train_groups = load_bearings(train_file, base)
    val, val_groups = load_bearings(val_file, base)
    verify_parent(base, "B", train)
    verify_split(train, val)
    old, old_file, historical = historical_model(ROOT, inputs)
    verify_parent(old, "B", train)
    if (historical["train_sha256"] != "sha256:" + digest(train_file)
            or historical["validation_sha256"] != "sha256:" + digest(val_file)):
        raise ValueError("Historical reproduction data changed")
    original = base.predict(val.x)
    experiment_id = f"b-phase1-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    out = ROOT / "artifacts" / experiment_id
    out.mkdir(parents=True, exist_ok=False)
    common = {
        "study": experiment_id, "source_commit": BASE_COMMIT,
        "evaluation": "official_public_B_validation_only", "target_unit": "minutes",
        "train_sha256": digest(train_file), "validation_sha256": digest(val_file),
        "official_checkpoint_sha256": digest(base_file), "threads": 1,
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "llm_calls": 0, "pytrio_training_calls": 0,
    }
    registration = {
        **common, "dataset_manifest": declared,
        "swanlab_destination": PROJECT_PATH,
        "swanlab_mode_requested": "offline" if args.offline else "publish_and_verify",
        "train_groups": dict(sorted(Counter(train_groups.tolist()).items())),
        "validation_groups": dict(sorted(Counter(val_groups.tolist()).items())),
        "historical_inputs": inputs["historical_artifacts"],
        "code_sha256": {
            "runner": digest(Path(__file__)),
            "bearing_study": digest(ROOT / "src/zhixun/bearing_study.py"),
            "bearing_swanlab": digest(ROOT / "src/zhixun/bearing_swanlab.py"),
            "vendor_model": digest(ROOT / "src/zhixun/vendor/model.py"),
        },
    }
    write_json(out / "registration.json", registration)
    print("STUDY:", out, flush=True)
    runs = []

    def arm(name, parent, parent_sha, **options):
        result = run_arm(out, name, parent, train, val, val_groups, **options)
        result["parent_sha256"] = parent_sha
        runs.append(result)
        after = result["after"]
        print(f"{name}: RMSE={after['overall']['rmse']:.6f} MAE={after['overall']['mae']:.6f} "
              f"avg={after['bearing_mean_rmse']:.6f} worst={after['bearing_worst_rmse']:.6f} "
              f"epochs={result['completed_epochs']} cpu={result['training_cpu_seconds']:.3f}s", flush=True)
        return result

    official_sha = common["official_checkpoint_sha256"]
    static = arm("B0-static", base, official_sha, epochs=0)
    control = arm("B1-standard", base, official_sha, epochs=5, rate=.001)
    weights = bearing_weights(train_groups, .5)
    balanced = arm("B2-balanced", base, official_sha, epochs=5, rate=.001, weights=weights)
    reproduction = arm("B3-historical", old, digest(old_file), epochs=40, rate=.003)
    reproduction["historical_reproduction"] = {
        "reference_rmse": historical["validation_after"]["rmse"],
        "reference_mae": historical["validation_after"]["mae"],
        "matches_metrics": all(
            math.isclose(reproduction["after"]["overall"][k], historical["validation_after"][k],
                         rel_tol=0, abs_tol=1e-8) for k in ("rmse", "mae")),
        "exact_checkpoint_match": ("sha256:" + reproduction["checkpoint_sha256"]
                                   == historical["checkpoint_sha256"]),
        "legacy_migration_max_prediction_delta": float(np.max(np.abs(old.predict(val.x) - original))),
    }

    capacity = None
    if reproduction["historical_reproduction"]["matches_metrics"]:
        narrow = arm("B3-24-paired", base, official_sha, epochs=40, rate=.003)
        expanded = expand_hidden(base, 96)
        max_delta = float(np.max(np.abs(expanded.predict(val.x) - original)))
        if max_delta > 1e-9:
            raise ValueError("New widening should preserve the parent's public predictions")
        start_file = out / "migrated-from-official-96.npz"
        expanded.save(start_file)
        wide = arm("B3-96-paired", expanded, digest(start_file), epochs=40, rate=.003)
        wide["migration_origin_sha256"] = official_sha
        target = wide["training_cpu_seconds"]
        timed = arm("B3-24-near-cpu", base, official_sha, epochs=240, rate=.003, cpu_budget=target)
        capacity = {
            "migration_max_prediction_delta": max_delta,
            "wide_vs_equal_epochs": bearing_gate(narrow["after"], wide["after"]),
            "wide_vs_near_cpu": bearing_gate(timed["after"], wide["after"]),
            "cpu_ratio_narrow_to_wide": timed["training_cpu_seconds"] / target,
            "within_20_percent_cpu": abs(timed["training_cpu_seconds"] / target - 1) <= .2,
        }

    for run in runs:
        if run is not static:
            run["gate_vs_B0"] = bearing_gate(static["after"], run["after"])
        if run is not static and run is not control:
            run["gate_vs_B1"] = bearing_gate(control["after"], run["after"])
        write_json(out / run["run"] / "result.json", run)
    verify_inputs("B", ROOT)
    decision = {
        "B2_passed": balanced["gate_vs_B1"]["passes"],
        "B3_reproduced": reproduction["historical_reproduction"]["matches_metrics"],
        "widened_passed_matched_checks": bool(
            capacity and capacity["within_20_percent_cpu"]
            and capacity["wide_vs_equal_epochs"]["passes"] and capacity["wide_vs_near_cpu"]["passes"]),
        "stage": "development_only",
    }
    summary = {"registration": registration, "runs": runs, "capacity": capacity, "decision": decision}
    write_metric_tables(out, runs)
    write_json(out / "summary.json", summary)
    for run in runs:
        run["swanlab"] = record_swanlab(out, run, common)
        write_json(out / run["run"] / "result.json", run)
        write_json(out / "summary.json", summary)
    failed = publish_study(out, summary) if not args.offline else 0
    print("DECISION:", json.dumps(decision), flush=True)
    print("RESULTS:", out / "summary.json", flush=True)
    if failed:
        print(f"{failed} local SwanLab runs still need cloud verification. "
              f'Use --publish-only "{out}" to retry without retraining.', file=sys.stderr)
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())

"""Run paired native C continuation arms on the official public data."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
from datetime import UTC, datetime
from time import perf_counter, process_time
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for variable in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from zhixun.bearing_swanlab import PROJECT, WORKSPACE
from zhixun.c_model import c_metrics, condition_residual, engine_weights, fit_epoch
from zhixun.contest_runtime import predict
from zhixun.data import digest, load_dataset, verify_inputs, verify_parent, verify_split, write_json
from zhixun.model_extension import expand_hidden
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model


def metadata(file: Path, ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    with file.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if tuple(item["id"] for item in rows) != ids:
        raise ValueError("Official C metadata order differs from the feature matrix")
    items = [row["meta"] for row in rows]
    return np.asarray([r["source_subset"] for r in items]), np.asarray([
        f"{r['source_subset']}/{r['unit']}" for r in items])


def evaluate(pred, valid, subset, engines, regimes):
    groups = {}
    for key, values in (("subset", subset), ("engine", engines), ("regime", regimes)):
        groups[key] = {str(g): c_metrics(pred[values == g], valid.y[values == g]) for g in np.unique(values)}
    groups["lifetime"] = {}
    for lower, upper in [(0, 50), (50, 100), (100, 200), (200, math.inf)]:
        m = (lower <= valid.y) & (valid.y < upper)
        groups["lifetime"][f"{lower}-{upper}"] = c_metrics(pred[m], valid.y[m])
    return {"overall": c_metrics(pred, valid.y), **groups,
            "mean_engine_rmse": float(np.mean([d["rmse"] for d in groups["engine"].values()])),
            "worst_engine_rmse": max(d["rmse"] for d in groups["engine"].values())}


def compare(baseline, candidate):
    old, new = baseline["overall"], candidate["overall"]
    near_old, near_new = baseline["lifetime"]["0-50"], candidate["lifetime"]["0-50"]
    checks = {"rmse_down": new["rmse"] < old["rmse"], "mae_not_up": new["mae"] <= old["mae"],
              "nasa_reference_not_up": new["nasa_reference_sum"] <= old["nasa_reference_sum"],
              "device_mean_rmse_not_up": candidate["mean_engine_rmse"] <= baseline["mean_engine_rmse"],
              "near_failure_rmse_not_up": near_new["rmse"] <= near_old["rmse"],
              "near_failure_overestimate_p95_not_up": near_new["overestimate_p95"] <= near_old["overestimate_p95"]}
    return {"pass": all(checks.values()), "checks": checks}


def record(folder, config, metrics, losses):
    recording = Recorder(folder, config, mode="offline", project=PROJECT, workspace=WORKSPACE)
    try:
        for i, loss in enumerate(losses, 1):
            recording.log(i, {"training/objective": loss})
        numeric = {f"validation/{name}": value for name, value in metrics["overall"].items()}
        numeric.update({f"validation/near_failure_{name}": value
                        for name, value in metrics["lifetime"]["0-50"].items()})
        numeric.update({f"validation/{key}": metrics[key] for key in ("mean_engine_rmse", "worst_engine_rmse")})
        for kind in ("subset", "regime"):
            for group, outcome in metrics[kind].items():
                numeric[f"validation/{kind}/{group}/rmse"] = outcome["rmse"]
        recording.log(len(losses) + 1, numeric)
    except Exception:
        recording.finish(failed=True)
        raise
    recording.finish()
    return recording.reference()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.home() / "Desktop/zhixun-project")
    args = parser.parse_args()
    source = args.source_root.resolve() / "data/C"
    provenance = verify_inputs("C", args.source_root)
    initial = Model.load(source / "checkpoint.npz")
    train = load_dataset(source / "train.jsonl", initial, "C")
    valid = load_dataset(source / "validation.jsonl", initial, "C")
    verify_parent(initial, "C", train)
    verify_split(train, valid)
    _, train_engine = metadata(source / "train.jsonl", train.ids)
    subset, device = metadata(source / "validation.jsonl", valid.ids)
    setting = initial.meta["paths"].index('[["key","settings"],["index",0]]')
    train_regimes, valid_regimes = (np.rint(rows.x[:, setting]).astype(int) for rows in (train, valid))
    weights = engine_weights(train_engine, train_regimes)
    if set(train_regimes) != set(valid_regimes):
        raise ValueError("C validation observed a new operating regime")
    experiment = ROOT / "artifacts" / f"c-study-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:6]}"
    experiment.mkdir(parents=True)
    plans = [
        {"name": "C0-official", "recipe": "start", "epochs": 0},
        {"name": "C1-adam5", "recipe": "current_round", "epochs": 5, "initial_lr": .001, "end_lr": .001},
        {"name": "C2-wide-mse", "recipe": "widen_mse", "epochs": 50, "initial_lr": .003, "end_lr": .0004},
        {"name": "C3-cond-ridge", "recipe": "paired_tanh_ridge", "epochs": 0},
        {"name": "C4-cond-adam", "recipe": "paired_tanh_mse", "epochs": 35, "initial_lr": .0015, "end_lr": .0002},
        {"name": "C5-cond-tail", "recipe": "paired_tanh_asymmetric", "epochs": 35, "initial_lr": .0015,
         "end_lr": .0002, "near_weight": 1.5, "tail_weight": .18},
    ]
    registration = {
        "study": experiment.name,
        "source_archive": provenance["archive"],
        "source_archive_sha256": provenance["archive_sha256"],
        "label_status_declared": provenance["label_status_declared"],
        "source_commit": "271263719f06c6b684a30c4d5421035b06e93369",
        "train_sha256": digest(source / "train.jsonl"),
        "validation_sha256": digest(source / "validation.jsonl"),
        "parent_sha256": digest(source / "checkpoint.npz"),
        "source_sha256": digest(ROOT / "src/zhixun/c_model.py"),
        "runner_sha256": digest(Path(__file__)),
        "train_rows": len(train.y), "train_engines": len(train.groups),
        "validation_rows": len(valid.y), "validation_engines": len(valid.groups),
        "operating_conditions": sorted(set(train_regimes.tolist())), "plans": plans,
        "scope": "official_public_c_development",
        "metrics": {"NASA": "offline reference sum: expm1(e/10) when e>0; expm1(-e/13) otherwise; e=pred-target",
                    "near_failure": "real RUL in [0,50) cycles"},
    }
    write_json(experiment / "registration.json", registration)
    print("STUDY", experiment, flush=True)
    summary = {"registration": registration, "runs": []}
    ridge = None
    ridge_cpu = 0.0
    baseline = None
    for plan in plans:
        start_cpu, start_wall = process_time(), perf_counter()
        losses = []
        migration_delta = 0.0
        recipe = plan["recipe"]
        model = copy.deepcopy(initial)
        if recipe == "widen_mse":
            model = expand_hidden(initial, 96)
            migration_delta = float(np.max(np.abs(model.predict(train.x[:2000]) - initial.predict(train.x[:2000]))))
        elif recipe == "paired_tanh_ridge":
            ridge, migration_delta = condition_residual(initial, train.x, train.y, weights)
            ridge_cpu = process_time() - start_cpu
            model = copy.deepcopy(ridge)
        elif recipe in ("paired_tanh_mse", "paired_tanh_asymmetric"):
            if ridge is None:
                raise ValueError("Conditions were never fitted")
            model = copy.deepcopy(ridge)
        if migration_delta > 1e-9:
            raise ValueError("Migration failed to preserve official start")
        for epoch in range(plan["epochs"]):
            fraction = epoch / max(1, plan["epochs"] - 1)
            factor = (1 + math.cos(math.pi * fraction)) / 2
            model.meta["learning_rate"] = plan["end_lr"] + (plan["initial_lr"] - plan["end_lr"]) * factor
            if recipe in ("current_round", "widen_mse"):
                losses.extend(model.fit(train.x, train.y, 1))
            else:
                losses.append(fit_epoch(model, train.x, train.y, weights,
                                        near_weight=plan.get("near_weight", 1.), tail_weight=plan.get("tail_weight", 0.)))
            if epoch % 10 == 9:
                print(plan["name"], "epoch", epoch + 1, flush=True)
        cpu, wall = process_time() - start_cpu, perf_counter() - start_wall
        outcome = evaluate(model.predict(valid.x), valid, subset, device, valid_regimes)
        folder = experiment / plan["name"]
        folder.mkdir()
        filename = folder / "checkpoint.npz"
        model.save(filename)
        assert np.allclose(model.predict(valid.x), Model.load(filename).predict(valid.x), rtol=0, atol=1e-12)
        details = {
            "name": plan["name"], "plan": plan, "validation": outcome,
            "training_cpu_seconds": cpu, "training_wall_seconds": wall,
            "prerequisite_ridge_cpu_seconds": ridge_cpu if recipe in ("paired_tanh_mse", "paired_tanh_asymmetric") else 0.,
            "hidden": model.meta["hidden"], "migration_max_difference": migration_delta,
            "checkpoint_sha256": digest(filename), "completed_epochs": model.meta["epochs"] - initial.meta["epochs"],
            "optimizer_steps_added": model.meta["steps"] - initial.meta["steps"],
        }
        config = {**{key: val for key, val in registration.items() if key != "plans"},
                  **plan, "checkpoint_sha256": details["checkpoint_sha256"], "hidden": model.meta["hidden"]}
        details["swanlab"] = {**record(folder, config, outcome, losses), "configuration": config}
        if baseline is not None:
            details["gate_vs_start"] = compare(baseline, outcome)
            if len(summary["runs"]) >= 2:
                details["gate_vs_control"] = compare(summary["runs"][1]["validation"], outcome)
        else:
            baseline = outcome
        write_json(folder / "result.json", details)
        summary["runs"].append(details)
        write_json(experiment / "summary.json", summary)
        print(plan["name"], outcome["overall"], outcome["lifetime"]["0-50"], flush=True)

    eligible = [r for r in summary["runs"][2:] if r["gate_vs_control"]["pass"] and r["gate_vs_start"]["pass"]]
    if eligible:
        best = min(eligible, key=lambda r: r["validation"]["overall"]["rmse"])
        folder = experiment / best["name"]
        raw = Model.load(folder / "checkpoint.npz").predict(valid.x)
        os.environ.pop("CONTEST_ARTIFACT_PATH", None)
        with (source / "validation.jsonl").open(encoding="utf-8") as stream:
            outputs = [predict(SimpleNamespace(task="regression", inputs=json.loads(line)["inputs"]), folder)["value"]
                       for line in stream if line.strip()]
        parity = float(np.max(np.abs(raw - outputs)))
        if len(outputs) != len(valid.y) or parity >= 1e-9:
            raise ValueError("Frozen prediction compatibility verification failed")
        summary["selection"] = {"name": best["name"], "rows_verified": len(outputs), "max_predict_delta": parity,
                                "checkpoint_sha256": best["checkpoint_sha256"], "rule": "best rmse among dual-gate passes"}
    else:
        summary["selection"] = {"name": None, "rule": "no candidate passed against both start and five-epoch control"}
    write_json(experiment / "summary.json", summary)
    print("RESULT", experiment / "summary.json", summary["selection"], flush=True)


if __name__ == "__main__":
    main()

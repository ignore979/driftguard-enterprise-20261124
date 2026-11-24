"""B study: leave-one-training-bearing-out tabular teachers and continued MLP students."""

from __future__ import annotations

import copy
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for variable in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from zhixun.bearing_study import bearing_gate, bearing_metrics, bearing_weights, fit_weighted_bearing_epoch, load_bearings
from zhixun.data import Dataset, digest, verify_inputs, verify_parent, verify_split, write_json
from zhixun.model_extension import expand_hidden
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model


TEACHERS = (
    ("ridge-vibration", 0, 1.0, .15),
    ("ridge-rbf-local", 384, .33, .15),
    ("ridge-rbf-broad", 384, .13, .15),
)
STUDENTS = (
    ("24-control", 24, .0), ("24-distilled", 24, .4),
    ("96-control", 96, .0), ("96-distilled", 96, .4),
)
WORKSPACE = "25165030"
PROJECT = "zhixun-school_from_813_1794808892434"
PREVIOUS = ROOT / "artifacts/b-phase1-20261123T142953-3f0ddb42/B2-balanced/checkpoint.npz"


def vibration_features(x: np.ndarray) -> np.ndarray:
    # Both channels follow the official alphabetical paths. The three band
    # powers also have stable relative meanings under a change of gain.
    halves = [x[:, :17], x[:, 17:]]
    relative = []
    for channel in halves:
        energy = np.maximum(channel[:, :3], 0)
        relative.append(np.log((energy + 1e-10) / (energy.sum(axis=1, keepdims=True) + 3e-10)))
    for pos in (0, 1, 2, 11):
        relative.append(np.log((np.maximum(halves[0][:, pos:pos+1], 0) + 1e-10)
                               / (np.maximum(halves[1][:, pos:pos+1], 0) + 1e-10)))
    return np.column_stack((x, *relative))


def teacher_predict(x, y, group, query, config):
    _, features, bandwidth, penalty = config
    x = vibration_features(x)
    query = vibration_features(query)
    raw_scale = np.maximum(np.median(np.abs(x), axis=0), 1e-10)
    x = np.arcsinh(x / raw_scale)
    query = np.arcsinh(query / raw_scale)
    low, mid, high = np.quantile(x, [.05, .5, .95], axis=0)
    scale = np.maximum(high - low, .01)
    x = np.clip(2.0 * (x - mid) / scale, -3, 3)
    query = np.clip(2.0 * (query - mid) / scale, -3, 3)
    if features:
        rng = np.random.default_rng(4522)
        directions = rng.normal(0, bandwidth, size=(x.shape[1], features))
        phases = rng.uniform(0, 2 * np.pi, features)
        x = np.column_stack((x, np.sqrt(2) * np.cos(x @ directions + phases)))
        query = np.column_stack((query, np.sqrt(2) * np.cos(query @ directions + phases)))
    x = np.column_stack((x, np.ones(len(x))))
    query = np.column_stack((query, np.ones(len(query))))
    w = bearing_weights(group, .5)
    wx = x * w[:, None]
    gram = wx.T @ x
    gram.flat[::len(gram) + 1] += len(x) * penalty
    gram[-1, -1] -= len(x) * penalty  # Allow the intercept to fit the training mean.
    coefficients = np.linalg.solve(gram, wx.T @ y)
    return np.clip(query @ coefficients, 0, np.max(y))


def train_student(parent, data, groups, target, width, mix):
    model = copy.deepcopy(parent) if width == parent.meta["hidden"] else expand_hidden(parent, width)
    assert np.allclose(model.predict(data.x[:50]), parent.predict(data.x[:50]), rtol=0, atol=1e-9)
    objective = Dataset(data.ids, data.x, (1 - mix) * data.y + mix * target, data.groups)
    weights = bearing_weights(groups, .5)
    losses = []
    for stage in range(60):
        model.meta["learning_rate"] = .002 if stage < 40 else .0006
        losses.append(fit_weighted_bearing_epoch(model, objective, weights))
    return model, losses


def show(name, metrics):
    overall = metrics["overall"]
    print(f"{name}: RMSE {overall['rmse']:.4f}, MAE {overall['mae']:.4f}, "
          f"equal {metrics['bearing_mean_rmse']:.4f}, groups "
          f"{[round(v['rmse'], 4) for v in metrics['bearings'].values()]}", flush=True)


def record(path, result, common):
    fields = dict(result.get("configuration", {}))
    fields["role"] = result["role"]
    cfg = {**common, **fields, "run": path.name}
    recorder = Recorder(path, cfg, mode="offline", project=PROJECT, workspace=WORKSPACE)
    try:
        losses = result.get("losses", [])
        for step, loss in enumerate(losses, 1):
            recorder.log(step, {"training/objective": loss})
        metrics = {}
        for split, key in (("train_lobo", "training_lobo"), ("validation", "validation")):
            if key not in result:
                continue
            measured = result[key]
            metrics.update({f"{split}/{k}": float(v) for k, v in measured["overall"].items() if v is not None})
            metrics[f"{split}/bearing_mean_rmse"] = measured["bearing_mean_rmse"]
            metrics[f"{split}/bearing_worst_rmse"] = measured["bearing_worst_rmse"]
            for name, device in measured["bearings"].items():
                metrics[f"{split}/bearing/{name}/rmse"] = device["rmse"]
        metrics["training/cpu_seconds"] = result["training_cpu_seconds"]
        metrics["training/wall_seconds"] = result["training_wall_seconds"]
        recorder.log(len(losses)+1, metrics)
    except Exception:
        recorder.finish(failed=True)
        raise
    recorder.finish()
    return {**recorder.reference(), "configuration": cfg, "directory": str(path / "swanlog")}


def main():
    verify_inputs("B", ROOT)
    parent_file = ROOT / "data/B/checkpoint.npz"
    model = Model.load(parent_file)
    train_file, val_file = ROOT / "data/B/train.jsonl", ROOT / "data/B/validation.jsonl"
    train, tg = load_bearings(train_file, model)
    val, vg = load_bearings(val_file, model)
    verify_parent(model, "B", train)
    verify_split(train, val)
    expected = [json.loads(p)[-1][1] for p in model.meta["paths"]]
    assert all(expected[base:base+3] == ["band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k"]
               and expected[base+11] == "rms" for base in (0, 17))
    out = ROOT / "artifacts" / f"b-residual-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    common = {"study": out.name, "train_sha256": digest(train_file), "validation_sha256": digest(val_file),
              "parent_sha256": digest(parent_file), "source_sha256": digest(Path(__file__)),
              "public_scope": "official_public_B_validation_only", "target_unit": "minutes", "threads": 1}
    write_json(out / "registration.json", {**common, "teachers": TEACHERS, "students": STUDENTS,
                                           "selection": "training bearing LOBO mean RMSE + 0.25*worst",
                                           "teacher_candidates_are_research_only": True})
    print("STUDY", out, flush=True)
    before = bearing_metrics(Model.load(PREVIOUS).predict(val.x), val.y, vg)
    results = []
    oof_predictions = []
    for config in TEACHERS:
        cpu, wall = process_time(), perf_counter()
        oof = np.empty_like(train.y)
        for group in np.unique(tg):
            fold = tg == group
            oof[fold] = teacher_predict(train.x[~fold], train.y[~fold], tg[~fold], train.x[fold], config)
        lobo = bearing_metrics(oof, train.y, tg)
        results.append({"name": f"teacher-{config[0]}", "role": "research_diagnostic", "training_lobo": lobo,
                        "selection_score": lobo["bearing_mean_rmse"] + .25 * lobo["bearing_worst_rmse"],
                        "configuration": dict(zip(("teacher", "rff", "bandwidth", "ridge_penalty"), config)),
                        "training_cpu_seconds": process_time()-cpu, "training_wall_seconds": perf_counter()-wall})
        oof_predictions.append(oof)
        show(f"LOBO {config[0]}", lobo)
    selected = min(range(len(TEACHERS)), key=lambda i: results[i]["selection_score"])
    write_json(out / "training_selection.json", {"teacher": TEACHERS[selected][0],
                                                  "teacher_scores": {r["name"]: r["selection_score"] for r in results}})
    for config, result in zip(TEACHERS, results, strict=True):
        cpu, wall = process_time(), perf_counter()
        prediction = teacher_predict(train.x, train.y, tg, val.x, config)
        result["training_cpu_seconds"] += process_time()-cpu
        result["training_wall_seconds"] += perf_counter()-wall
        result["validation"] = bearing_metrics(prediction, val.y, vg)
        show(result["name"], result["validation"])
    for name, width, mix in STUDENTS:
        cpu, wall = process_time(), perf_counter()
        student, losses = train_student(model, train, tg, oof_predictions[selected], width, mix)
        result = {"name": f"student-{name}", "role": "official_continuation_development",
                  "configuration": {"width": width, "teacher_mix": mix, "teacher": TEACHERS[selected][0],
                                    "epochs": 60, "sample_weight_bearing_fraction": .5,
                                    "rate_first_40": .002, "rate_last_20": .0006},
                  "training_cpu_seconds": process_time()-cpu, "training_wall_seconds": perf_counter()-wall,
                  "losses": losses}
        result["validation"] = bearing_metrics(student.predict(val.x), val.y, vg)
        result["gate_vs_B2"] = bearing_gate(before, result["validation"])
        folder = out / result["name"]
        folder.mkdir()
        checkpoint = folder / "checkpoint.npz"
        student.save(checkpoint)
        assert np.allclose(Model.load(checkpoint).predict(val.x), student.predict(val.x), rtol=0, atol=1e-9)
        result["checkpoint_sha256"] = digest(checkpoint)
        results.append(result)
        show(result["name"], result["validation"])
    summary = {"registration": common, "teacher_selected_on_training": TEACHERS[selected][0],
               "B2": before, "runs": results}
    write_json(out / "summary.json", summary)
    for result in results:
        folder = out / result["name"]
        folder.mkdir(exist_ok=True)
        result["swanlab"] = record(folder, result, common)
        write_json(folder / "result.json", result)
        write_json(out / "summary.json", summary)
    print("RESULTS", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()

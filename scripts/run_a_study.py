"""Compare two bounded A continuation policies on the unchanged public A validation."""

from __future__ import annotations

import argparse
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
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

from zhixun.bearing_swanlab import PROJECT, WORKSPACE
from zhixun.data import digest, load_dataset, verify_inputs, verify_parent, verify_split, write_json
from zhixun.metrics import promotion, score
from zhixun.model_extension import ClassWeightedModel
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model

ARMS = {"A-cw05": 0.5, "A-cw15": 1.5}
SCHEDULE = (
    (1, .0005), (1, .0005), (1, .0005), (2, .0007), (3, .0007),
    (5, .001), (7, .001), (10, .001), (5, .0005), (10, .0005),
)
SOURCE_COMMIT = "271263719f06c6b684a30c4d5421035b06e93369"


def metrics_and_classes(model, validation):
    metrics, classes, _ = score(model, validation)
    return metrics, classes


def logged_metrics(prefix, metrics, details, classes):
    records = {f"{prefix}/{name}": value for name, value in metrics.items()}
    for label in classes:
        for name in ("support", "precision", "recall", "f1"):
            records[f"{prefix}/class_{label}/{name}"] = details[label][name]
    return records


def run_arm(folder, profile, config, parent_path, train, validation):
    folder.mkdir()
    selected = Model.load(parent_path)
    names = selected.meta["classes"]
    before, details_before = metrics_and_classes(selected, validation)
    baseline = {"metrics": before, "per_class": details_before}
    selected_round = 0
    selected_file = parent_path
    opened = Recorder(folder, config, mode="offline", project=PROJECT, workspace=WORKSPACE)
    rounds = []
    try:
        opened.log(0, logged_metrics("selected", before, details_before, names))
        for round_index, (epochs, rate) in enumerate(SCHEDULE, 1):
            start = digest(selected_file)
            candidate = copy.deepcopy(selected)
            candidate.__class__ = ClassWeightedModel
            candidate.meta["learning_rate"] = rate
            cpu_start = process_time()
            wall_start = perf_counter()
            losses = candidate.fit(train.x, train.y, epochs, cw_power=profile)
            train_cpu = process_time() - cpu_start
            train_wall = perf_counter() - wall_start
            metrics, details = metrics_and_classes(candidate, validation)
            eligible, rule = promotion(before, metrics, "A")
            candidate_file = folder / f"r{round_index:02d}.npz"
            candidate.save(candidate_file)
            round_result = {
                "round": round_index,
                "parent_sha256": start,
                "parent_metrics": before,
                "epochs": len(losses),
                "learning_rate": rate,
                "optimizer_steps_added": candidate.meta["steps"] - selected.meta["steps"],
                "train_cpu_seconds": train_cpu,
                "train_wall_seconds": train_wall,
                "candidate_checkpoint_sha256": digest(candidate_file),
                "candidate_metrics": metrics,
                "candidate_per_class": details,
                "losses": losses,
                "adopted_candidate": bool(eligible),
                "guard": rule,
            }
            if eligible:
                selected_file = candidate_file
                selected_round = round_index
                before, details_before = metrics, details
                selected = candidate
            round_result.update(selected_round=selected_round, selected_sha256=digest(selected_file))
            rounds.append(round_result)
            opened.log(round_index, {
                **logged_metrics("candidate", metrics, details, names),
                **logged_metrics("selected", before, details_before, names),
                "decision/adopted": float(eligible),
                "training/epochs": len(losses),
                "training/optimizer_steps": round_result["optimizer_steps_added"],
                "training/loss": losses[-1],
                "training/learning_rate": rate,
                "training/cpu_seconds": train_cpu,
                "training/wall_seconds": train_wall,
            })
            print(f"{folder.name} r{round_index}: {metrics['accuracy']:.6f} accuracy, "
                  f"{metrics['macro_f1']:.6f} Macro-F1, selected r{selected_round}", flush=True)
    except BaseException:
        opened.finish(failed=True)
        raise
    opened.finish()
    final_file = folder / "checkpoint.npz"
    selected.save(final_file)
    reloaded = Model.load(final_file)
    final_metrics, final_details = metrics_and_classes(reloaded, validation)
    if final_metrics != before or final_details != details_before:
        raise ValueError("Saved A checkpoint results differ from the selected model")
    result = {
        "arm": folder.name,
        "scope": "official_public_fixed_a_validation_local",
        "baseline": baseline,
        "after": {"metrics": final_metrics, "per_class": final_details},
        "parent_sha256": digest(parent_path),
        "checkpoint_sha256": digest(final_file),
        "selected_version_sha256": digest(selected_file),
        "selected_round": selected_round,
        "adopted_rounds": sum(r["adopted_candidate"] for r in rounds),
        "training_cpu_seconds": sum(r["train_cpu_seconds"] for r in rounds),
        "training_wall_seconds": sum(r["train_wall_seconds"] for r in rounds),
        "rounds": rounds,
        "swanlab": {**opened.reference(), "configuration": config},
    }
    write_json(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "Desktop" / "zhixun-project")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    source = verify_inputs("A", data_root)
    train_file = data_root / "data/A/train.jsonl"
    validation_file = data_root / "data/A/validation.jsonl"
    parent_file = data_root / "data/A/checkpoint.npz"
    parent = Model.load(parent_file)
    train = load_dataset(train_file, parent, "A")
    verify_parent(parent, "A", train)
    study = ROOT / "artifacts" / f"a-study-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:6]}"
    study.mkdir(parents=True, exist_ok=False)
    registration = {
        "study": study.name,
        "source_commit": SOURCE_COMMIT,
        "scope": "official_public_fixed_a_validation_local",
        "data_archive": source["archive"],
        "source_label_status": source["label_status_declared"],
        "train_sha256": digest(train_file),
        "validation_sha256": digest(validation_file),
        "parent_sha256": digest(parent_file),
        "script_sha256": digest(Path(__file__)),
        "train_rows": len(train.y),
        "epochs_lr_by_round": [list(spec) for spec in SCHEDULE],
        "arms": ARMS,
        "selection": "Same-panel Macro-F1 must increase and accuracy must not decrease",
    }
    write_json(study / "registration.json", registration)
    print(f"A study {study}", flush=True)
    validation = load_dataset(validation_file, parent, "A")
    verify_split(train, validation)
    results = []
    for arm, power in ARMS.items():
        config = {**registration, "arm": arm, "cw_power": power, "validation_rows": len(validation.y)}
        results.append(run_arm(study / arm, power, config, parent_file, train, validation))
        write_json(study / "summary.json", {"registration": registration, "runs": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

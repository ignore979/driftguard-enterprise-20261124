"""Replay ten bounded C continuations over the official public projection."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time
from types import SimpleNamespace
from uuid import uuid4

from run_c_model import PROJECT, ROOT, WORKSPACE, evaluate, metadata

import numpy as np

from zhixun.c_round_model import C_STRATEGIES, OPERATING_SETTING, continue_c, promote_c
from zhixun.contest_runtime import predict, rows_dataset
from zhixun.data import digest, load_dataset, verify_inputs, verify_parent, verify_split, write_json
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model

SCHEDULE = ((8, .001), (8, .0007), (8, .001), (8, .0005), (8, .0007),
            (5, .0005), (5, .00035), (5, .00025), (5, .00015), (5, .0001))


def projection(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            yield {key: row[key] for key in ("id", "inputs", "answer")}


def runtime_parity(val_file, selected, expected):
    with val_file.open(encoding="utf-8") as stream:
        values = [
            predict(SimpleNamespace(task="regression", inputs=json.loads(line)["inputs"]), selected.parent)["value"]
            for line in stream
        ]
    difference = float(np.max(np.abs(expected - values)))
    if difference > 1e-9:
        raise ValueError("C frozen predictor diverged from batch inference")
    return difference


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-root", type=Path, default=Path.home() / "Desktop/zhixun-project")
    cli.add_argument("--strategy", choices=sorted(C_STRATEGIES), default="condition")
    args = cli.parse_args()
    source = args.source_root.resolve() / "data/C"
    provenance = verify_inputs("C", args.source_root)
    initial_file, train_file, val_file = (source / name for name in ("checkpoint.npz", "train.jsonl", "validation.jsonl"))
    initial = Model.load(initial_file)
    full_train, full_val = (load_dataset(p, initial, "C") for p in (train_file, val_file))
    verify_parent(initial, "C", full_train)
    verify_split(full_train, full_val)
    train, val = (rows_dataset(projection(p), initial) for p in (train_file, val_file))
    if any(not (actual.ids == reference.ids and np.array_equal(actual.x, reference.x)
                and np.array_equal(actual.y, reference.y))
           for actual, reference in ((train, full_train), (val, full_val))):
        raise ValueError("Official projection differs from the source feature matrix")
    subset, device = metadata(val_file, val.ids)
    regime = np.rint(val.x[:, initial.meta["paths"].index(OPERATING_SETTING)]).astype(int)
    del full_train, full_val

    exp = ROOT / "artifacts" / f"c-study-rounds-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:4]}"
    arm = exp / f"C10-{args.strategy}"
    arm.mkdir(parents=True)
    config = {
        "study": exp.name, "strategy": args.strategy, "schedule": [list(i) for i in SCHEDULE],
        "source_commit": "271263719f06c6b684a30c4d5421035b06e93369",
        "train_sha256": provenance["files"]["train.jsonl"]["sha256"],
        "validation_sha256": provenance["files"]["validation.jsonl"]["sha256"],
        "parent_sha256": digest(initial_file), "train_rows": len(train.y), "validation_rows": len(val.y),
        "training_features": "SDK projection inputs only, no unit or cycle metadata",
        "evaluation_scope": "official_public_C_replay",
        "source_sha256": digest(ROOT / "src/zhixun/c_round_model.py"),
        "runner_sha256": digest(Path(__file__)),
    }
    write_json(exp / "registration.json", config)
    print("STUDY", exp, flush=True)
    parent = initial
    selected = initial_file
    baseline = evaluate(parent.predict(val.x), val, subset, device, regime)
    chosen = baseline
    result = {"name": arm.name, "configuration": config, "baseline": baseline, "rounds": []}
    log = Recorder(arm, config, mode="offline", project=PROJECT, workspace=WORKSPACE)
    try:
        champion_round = 0
        for i, (epochs, rate) in enumerate(SCHEDULE, 1):
            parent_file_before = selected
            start_cpu, start_wall = process_time(), perf_counter()
            model, losses, audit = continue_c(parent, train.x, train.y, epochs=epochs, learning_rate=rate,
                                             strategy=args.strategy)
            cpu, wall = process_time() - start_cpu, perf_counter() - start_wall
            metrics = evaluate(model.predict(val.x), val, subset, device, regime)
            gate = promote_c(baseline, chosen, metrics)
            folder = arm / f"r{i:02}"
            folder.mkdir()
            checkpoint = folder / "checkpoint.npz"
            model.save(checkpoint)
            if gate["pass"]:
                champion_round = i
                parent, chosen, selected = Model.load(checkpoint), metrics, checkpoint
            run = {"round": i, "candidate": metrics, "selection": gate, "selected_round": champion_round,
                   "recipe": audit, "epochs": epochs, "learning_rate": rate,
                   "training_loss": losses, "cpu_seconds": cpu, "wall_seconds": wall,
                   "parent_sha256": digest(parent_file_before), "candidate_sha256": digest(checkpoint),
                   "selected_sha256": digest(selected)}
            write_json(folder / "round.json", run)
            result["rounds"].append(run)
            stats = {"round/adopted": int(gate["pass"]), "round/champion": champion_round,
                     "training/cpu_seconds": cpu, "training/epochs": epochs, "training/loss": losses[-1]}
            for role, evaluation in (("candidate", metrics), ("selected", chosen)):
                stats.update({f"{role}/{m}": evaluation["overall"][m] for m in ("rmse", "mae", "nasa_reference_sum")})
                stats[f"{role}/near_failure_rmse"] = evaluation["lifetime"]["0-50"]["rmse"]
                stats[f"{role}/near_failure_overestimate_p95"] = evaluation["lifetime"]["0-50"]["overestimate_p95"]
            log.log(i, stats)
            print(arm.name, i, audit["migration"], "promote" if gate["pass"] else "retain",
                  f"rmse {metrics['overall']['rmse']:.6f}", f"selected {chosen['overall']['rmse']:.6f}", flush=True)
        parity = runtime_parity(val_file, selected, parent.predict(val.x))
        result.update(validation=chosen, selected_round=champion_round, selected_sha256=digest(selected),
                      selected_checkpoint=str(selected), runtime_rows_verified=len(val.y), runtime_max_delta=parity,
                      promotion_count=sum(r["selection"]["pass"] for r in result["rounds"]))
    except Exception:
        log.finish(failed=True)
        raise
    log.finish()
    result["swanlab"] = {**log.reference(), "configuration": config}
    write_json(arm / "result.json", result)
    write_json(exp / "summary.json", result)
    print("RESULT", exp / "summary.json", result["promotion_count"], chosen["overall"], flush=True)


if __name__ == "__main__":
    main()

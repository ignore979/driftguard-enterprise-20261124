"""Reproduce ten C rounds on the fixed organizer public train and validation split."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for key in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

from zhixun.c_round import Recipe, continue_c, measure_c, nasa_sum, quality_gate, round_recipe
from zhixun.data import digest, load_dataset, verify_inputs, verify_parent, verify_split, write_json
from zhixun.vendor.model import Model


def panel_metrics(model, data, subsets):
    predicted = model.predict(data.x)
    metrics = measure_c(predicted, data.y)
    metrics["nasa_proxy_sum"] = nasa_sum(metrics)
    metrics["subsets"] = {}
    for subset in sorted(set(subsets)):
        mask = subsets == subset
        group = measure_c(predicted[mask], data.y[mask])
        metrics["subsets"][subset] = {**group, "nasa_proxy_sum": nasa_sum(group)}
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=("legacy", "symmetric", "asymmetric"), required=True)
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("Repeated arm")
    official = args.official_root.resolve()
    declared = verify_inputs("C", official)
    files = official / "data/C"
    baseline_file = files / "checkpoint.npz"
    baseline = Model.load(baseline_file)
    training = load_dataset(files / "train.jsonl", baseline, "C")
    validation = load_dataset(files / "validation.jsonl", baseline, "C")
    verify_parent(baseline, "C", training)
    verify_split(training, validation)
    with (files / "validation.jsonl").open(encoding="utf-8") as stream:
        subsets = np.asarray([json.loads(line)["meta"]["source_subset"] for line in stream if line.strip()])
    study = ROOT / "artifacts" / f"c-replay-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:6]}"
    study.mkdir(parents=True)
    reg = {
        "grouping": "archived_official_engine_split",
        "scope": "local_public_C_not_platform_run",
        "source_commit": "271263719f06c6b684a30c4d5421035b06e93369",
        "source": declared,
        "code_sha256": {k: digest(ROOT / k) for k in ("scripts/run_c_replay.py", "src/zhixun/c_round.py")},
        "train_engines": len(training.groups), "validation_engines": len(validation.groups),
        "rounds": 10, "arms": args.arms,
        "controls": {"legacy": "historical requested schedule 5 epochs at 1,5,9; frozen local RMSE/MAE gate",
                     "symmetric": "wide MSE control with same round epoch/LR schedule",
                     "asymmetric": "wide asymmetric continuation with all local quality gates"},
    }
    write_json(study / "registration.json", reg)
    print("STUDY", study, flush=True)
    initial = panel_metrics(baseline, validation, subsets)
    summary = {"registration": reg, "baseline": initial, "arms": {}}
    for arm in args.arms:
        folder = study / f"C-{arm}"
        folder.mkdir()
        parent, parent_file, before, accepted = baseline, baseline_file, initial, []
        rounds = []
        for index in range(1, 11):
            path = folder / f"r{index:02d}"
            path.mkdir()
            start = time.process_time()
            recipe = round_recipe(index)
            if arm == "symmetric":
                recipe = Recipe("wide-mse", recipe.width, recipe.epochs, recipe.rate, 1., 1.)
            if arm == "legacy":
                recipe = Recipe("legacy-mse", baseline.meta["hidden"], 5, .001, 1., 1.)
            ran = arm != "legacy" or index in (1, 5, 9)
            if ran:
                candidate, losses, delta = continue_c(parent, training, recipe, time.monotonic() + 180.)
                after = panel_metrics(candidate, validation, subsets)
                eligible, violations = quality_gate(before, after)
                if arm == "legacy":
                    checks = {"rmse": after["rmse"] < before["rmse"], "mae": after["mae"] <= before["mae"]}
                    violations = [key for key, passed in checks.items() if not passed]
                    eligible = not violations
                checkpoint = path / "checkpoint.npz"
                candidate.save(checkpoint)
            else:
                after, losses, eligible, violations, checkpoint, delta = before, [], False, ["scheduled_hold"], None, 0.
            record = {
                "index": index, "recipe": asdict(recipe) if ran else "scheduled_hold",
                "local_public_before": before, "local_public_after": after, "eligible": eligible,
                "violations": violations, "selected": "candidate" if eligible else "parent",
                "parent_sha256": digest(parent_file), "candidate_sha256": digest(checkpoint) if checkpoint else None,
                "losses": losses, "cpu_seconds": time.process_time() - start, "migration_delta": delta,
            }
            if eligible:
                parent, parent_file, before = candidate, checkpoint, after
                accepted.append(index)
            record["selected_sha256"] = digest(parent_file)
            record["selected_metrics"] = before
            write_json(path / "round.json", record)
            rounds.append(record)
            print(arm, index, record["selected"], round(after["rmse"], 4), round(after["nasa_proxy_sum"], 2),
                  round(after["overestimate_p95"], 3), violations, flush=True)
        final = folder / "checkpoint.npz"
        parent.save(final)
        result = {"arm": arm, "selected_rounds": accepted, "initial": initial, "final": before,
                  "checkpoint_sha256": digest(final), "rounds": rounds}
        write_json(folder / "result.json", result)
        summary["arms"][arm] = {"folder": folder.name, "selected_rounds": accepted, "final": before}
        write_json(study / "summary.json", summary)
    print("COMPLETE", study, flush=True)


if __name__ == "__main__":
    main()

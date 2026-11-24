"""Build a real C decision corpus from disjoint engine groups in official training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for var in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np

from zhixun.c_round import measure_c, quality_gate
from zhixun.c_trio_protocol import select_messages
from zhixun.data import digest, load_dataset, verify_inputs, verify_parent, write_json
from zhixun.vendor.model import Model


STUDIES = {"training": (6, 0), "evaluation": (4, 20)}
RECIPES = (
    ("adam-target-relief-7", 2, .001, 7.0),
    ("adam-target-relief-3", 3, .001, 3.0),
    ("plain-adam", 4, .001, 0.0),
    ("adam-low-rate-target-relief", 2, .0003, 2.0),
)


def index_groups(path: Path, ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    grouped = defaultdict(list)
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if row["id"] != ids[index]:
            raise ValueError("C corpus index no longer matches official rows")
        meta = row["meta"]
        grouped[f'{meta["source_subset"]}/{meta["unit"]}'].append(index)
    return {key: np.array(rows, dtype=np.int64) for key, rows in grouped.items()}


def split_groups(groups: dict) -> dict[str, dict[str, list[str]]]:
    by_subset = defaultdict(list)
    for key in groups:
        by_subset[key.split("/", 1)[0]].append(key)
    split = {name: {} for name in STUDIES}
    for subset, keys in sorted(by_subset.items()):
        keys.sort(key=lambda key: hashlib.sha256(("C-policy-20261124/" + key).encode()).digest())
        boundary = int(len(keys) * .60)
        split["training"][subset], split["evaluation"][subset] = keys[:boundary], keys[boundary:]
    return split


def fold_rows(pools: dict, groups: dict, index: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    fit, held = [], []
    for keys in pools.values():
        fit.extend(key for offset, key in enumerate(keys) if (offset + index) % 4)
        held.extend(key for offset, key in enumerate(keys) if not (offset + index) % 4)
    if not fit or not held or set(fit) & set(held):
        raise ValueError("C shadow replay has invalid engine partitions")
    return np.sort(np.concatenate([groups[key] for key in fit])), np.sort(
        np.concatenate([groups[key] for key in held])), fit, held


def pack_example(dataset, round_index, before, after, eligible, violations, history, recipe):
    payload = {
        "dataset": dataset, "round": round_index, "parent": before, "candidate": after,
        "eligible": bool(eligible), "violations": list(violations), "host_vote": "unavailable",
        "recipe": recipe, "history": history[-3:],
    }
    action = "candidate" if eligible else "parent"
    if eligible:
        reason = "Measured C checks pass; RMSE is lower and the host has not rejected the candidate."
    else:
        reason = "At least one measured C check failed: " + ", ".join(violations) + "."
    completion = json.dumps({"select": action, "reason": reason}, ensure_ascii=False)
    return {"messages": select_messages(payload), "completion": completion}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_root", type=Path)
    args = parser.parse_args()
    source = verify_inputs("C", args.official_root)
    archive_root = args.official_root / "data/C"
    path = archive_root / "train.jsonl"
    official = Model.load(archive_root / "checkpoint.npz")
    data = load_dataset(path, official, "C")
    verify_parent(official, "C", data)
    groups = index_groups(path, data.ids)
    split = split_groups(groups)
    out = ROOT / "artifacts" / f"c-policy-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    out.mkdir()
    definitions = {
        "study": out.name,
        "scope": "project_derived_policy_labels_from_official_C_train_engine_disjoint_shadow_replays",
        "official_contest_labels": False,
        "used_public_validation_or_hidden": False,
        "provenance": source,
        "partitions": split,
        "recipes": RECIPES,
        "source_sha256": digest(Path(__file__)),
    }
    write_json(out / "registration.json", definitions)
    print("CORPUS", out, flush=True)
    summaries = {}
    for partition, (fold_count, seed_offset) in STUDIES.items():
        output, evidence = [], []
        for fold in range(fold_count):
            fit, held, fit_keys, held_keys = fold_rows(split[partition], groups, fold)
            x, y = data.x[fit], data.y[fit]
            vx, vy = data.x[held], data.y[held]
            meta = official.meta
            parent = Model("regression", meta["paths"], [], x, y, seed=meta["seed"] + seed_offset + fold,
                           hidden=meta["hidden"], learning_rate=meta["learning_rate"], batch_size=meta["batch_size"])
            parent.meta["profile"] = "C"
            parent.fit(x, y, meta["epochs"])
            history = []
            for stage in range(1, 5):
                recipe, epochs, rate, target_relief = RECIPES[(stage + fold - 1) % len(RECIPES)]
                candidate = copy.deepcopy(parent)
                candidate.meta["learning_rate"] = rate
                # Lower the fit target near failure as a fixed training-only overestimation countermeasure.
                adjusted = np.maximum(0., y - target_relief * np.maximum(0., 1 - y / 70.))
                candidate.fit(x, adjusted, epochs)
                before = measure_c(parent.predict(vx), vy)
                after = measure_c(candidate.predict(vx), vy)
                eligible, violations = quality_gate(before, after)
                round_index = fold + stage
                output.append(pack_example("C", round_index, before, after, eligible, violations, history, recipe))
                evidence.append({"fold": fold, "fit_engines": fit_keys, "held_engines": held_keys,
                                 "parents_fit_only_on_fit_engines": True, "recipe": recipe,
                                 "stage": stage, "eligible": bool(eligible), "violations": violations})
                if eligible:
                    parent = candidate
                history.append({"round": round_index, "recipe": recipe,
                                "selected": "candidate" if eligible else "parent",
                                "rmse": after["rmse"] if eligible else before["rmse"]})
        with (out / f"{partition}.jsonl").open("w", encoding="utf-8") as stream:
            for result in output:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        write_json(out / f"{partition}_evidence.json", evidence)
        summaries[partition] = {"rows": len(evidence), "eligible": sum(row["eligible"] for row in evidence),
                                "sha256": digest(out / f"{partition}.jsonl"),
                                "engines": sum(map(len, split[partition].values()))}
        print(partition, summaries[partition], flush=True)
    if any(summary["eligible"] in (0, summary["rows"]) for summary in summaries.values()):
        raise ValueError("A C policy split failed to exercise both genuine actions")
    write_json(out / "summary.json", summaries)


if __name__ == "__main__":
    main()

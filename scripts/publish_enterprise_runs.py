"""Sync and verify existing independent A/C and PyTRiO experiment journals."""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import swanlab
import yaml

from zhixun.bearing_swanlab import PROJECT, WORKSPACE, connect, publish_run
from zhixun.cloud import normalize_config
from zhixun.data import digest, write_json


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def replay_metrics(record: dict) -> dict:
    metrics = {"training/cpu_seconds": record["cpu_seconds"], "training/epochs": len(record["losses"]),
               "decision/adopted": float(record["selected"] == "candidate")}
    for role, key in (("candidate", "local_public_after"), ("selected", "selected_metrics")):
        value = record[key]
        metrics.update({f"{role}/{metric}": item for metric, item in value.items()
                        if isinstance(item, (int, float)) and not isinstance(item, bool)})
        metrics.update({f"{role}/{subset}/rmse": item["rmse"] for subset, item in value["subsets"].items()})
    return metrics


def materialize(arm: Path) -> tuple[dict, Path]:
    log_root = ROOT / "lab" / digest(arm / "result.json")[:16]
    logs = [(path, run) for path in (arm / "swanlog", log_root) for run in path.glob("run-*")
            if (run / "files/config.yaml").is_file() and any(run.glob("*.swanlab"))]
    if len(logs) > 1:
        raise ValueError(f"Multiple native journals for {arm.name}")
    if logs:
        path, run = logs[0]
        cfg = yaml.safe_load((run / "files/config.yaml").read_text(encoding="utf-8"))
        return normalize_config(cfg), path
    result = read(arm / "result.json")
    if "publish_metrics" in result:
        history = result["publish_metrics"]
        config = result["publish_configuration"]
    elif "rounds" in result and isinstance(result["rounds"], list) and "selected_rounds" in result:
        study = arm.parent
        registration = read(study / "registration.json")
        config = {"study": study.name, "arm": arm.name,
                  "source_commit": registration["source_commit"],
                  "protocol_sha256": digest(study / "registration.json"),
                  "train_sha256": registration["source"]["files"]["train.jsonl"]["sha256"],
                  "validation_sha256": registration["source"]["files"]["validation.jsonl"]["sha256"],
                  "selected_rounds": result["selected_rounds"], "checkpoint_sha256": result["checkpoint_sha256"],
                  "scope": registration["scope"]}
        history = [{"step": row["index"], "metrics": replay_metrics(row)} for row in result["rounds"]]
    else:
        raise ValueError(f"Missing native journal or publishable statistics for {arm.name}")
    text = "".join(json.dumps(row, allow_nan=False) + "\n" for row in history)
    history_file = arm / "history.jsonl"
    if history_file.exists() and history_file.read_text(encoding="utf-8") != text:
        raise ValueError(f"Local metrics changed for {arm.name}")
    history_file.write_text(text, encoding="utf-8")
    log_root.mkdir(parents=True, exist_ok=True)
    swanlab.init(project=PROJECT, workspace=WORKSPACE, name=arm.name, mode="offline", public=False,
                 log_dir=str(log_root), config=config)
    try:
        for row in history:
            swanlab.log(row["metrics"], step=row["step"])
    except BaseException:
        swanlab.finish(state="crashed")
        raise
    swanlab.finish(state="success")
    return config, log_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=5)
    args = parser.parse_args()
    root = (ROOT / "artifacts").resolve()
    queue, indices = [], {}
    for study in (path.resolve() for path in args.studies):
        if (study.parent != root or study in indices
                or not re.fullmatch(r"(?:a-study|c-replay|c-study|c-host|tri-c|c-deploy)[A-Za-z0-9-]+", study.name)):
            parser.error("Expected one or more distinct enterprise study directories under artifacts")
        index = study / "swanlab_index.json"
        indices[study] = read(index) if index.is_file() else {}
        arms = sorted(path.parent for path in study.glob("*/result.json"))
        if not arms:
            parser.error(f"No completed experiments under {study.name}")
        for arm in arms:
            if arm.is_symlink():
                parser.error("Experiment folder cannot be a symlink")
            config, logs = materialize(arm)
            queue.append((arm, config, logs))
    print(f"READY: {len(queue)} finished experiment journals", flush=True)
    connect()
    context = local()

    def sync(arm, config, log_root):
        if not hasattr(context, "api"):
            context.api = connect()
        return publish_run(ROOT, arm, config, context.api, log_root=log_root)

    failed = False
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = {executor.submit(sync, *spec): spec[0] for spec in queue}
        for future in as_completed(tasks):
            arm = tasks[future]
            try:
                receipt = future.result()
            except Exception as exc:
                receipt = {"status": "error", "failure": type(exc).__name__, "cloud_record_verified": False}
            failed |= not receipt["cloud_record_verified"]
            indices[arm.parent][arm.name] = receipt
            write_json(arm.parent / "swanlab_index.json", indices[arm.parent])
            print(arm.parent.name, arm.name, receipt["status"], receipt.get("cloud_url", ""), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

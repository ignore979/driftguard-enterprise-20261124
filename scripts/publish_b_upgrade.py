"""Materialize B upgrade journals and synchronize every completed arm by its stable ID."""

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
from zhixun.cloud import expected_points, normalize_config
from zhixun.data import digest, write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(arm: Path) -> tuple[dict, Path]:
    short_log_root = ROOT / "lab" / digest(arm / "result.json")[:16]
    logs = [
        (directory, run) for directory in (arm / "swanlog", short_log_root)
        for run in directory.glob("run-*")
        if (run / "files/config.yaml").is_file() and any(run.glob("*.swanlab"))
    ]
    if logs:
        if len(logs) != 1:
            raise ValueError(f"Ambiguous local SwanLab journal: {arm.name}")
        directory, run = logs[0]
        return normalize_config(yaml.safe_load((run / "files/config.yaml").read_text(encoding="utf-8"))), directory
    result = read_json(arm / "result.json")
    history_file = arm / "history.jsonl"
    lines = [json.loads(row) for row in history_file.read_text(encoding="utf-8").splitlines()]
    points = expected_points(lines)
    if "validation/rmse" not in points:
        after = result["after"]
        checks = {
            "validation/rmse": after["overall"]["rmse"],
            "validation/mae": after["overall"]["mae"],
            "validation/bearing_mean_rmse": after["bearing_mean_rmse"],
            "validation/bearing_worst_rmse": after["bearing_worst_rmse"],
            "training/cpu_seconds": result["training_cpu_seconds"],
            "decision/passed_B2_gate": float(result["gate_vs_B2"]["passes"]),
        }
        checks.update({f"bearing/{g}/rmse": v["rmse"] for g, v in after["bearings"].items()})
        checks.update({f"bearing/{g}/rul_correlation": v
                       for g, v in after.get("per_bearing_rul_correlation", {}).items() if v is not None})
        lines.append({"step": max(row["step"] for row in lines) + 1, "metrics": checks})
        with history_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(lines[-1], allow_nan=False) + "\n")
    registration = read_json(arm.parent / "registration.json")
    config = {
        "study": arm.parent.name, "arm": arm.name, "scope": "development_public_B",
        "source_commit": registration["source_commit"],
        "train_sha256": registration["train_sha256"],
        "validation_sha256": registration["validation_sha256"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "registered_protocol_sha256": digest(arm.parent / "registration.json"),
        "settings": {key: value for key, value in result.items() if key in {
            "hidden", "width", "epochs", "initial_lr", "completed_epochs", "optimizer_steps_added",
            "elapsed_steps", "learning_rate", "device_mass", "stage_mass", "order_penalty",
            "smooth_huber_delta_minutes", "adaptive_device_mass", "near_failure_penalty",
        }},
    }
    short_log_root.mkdir(parents=True, exist_ok=True)
    swanlab.init(project=PROJECT, workspace=WORKSPACE, name=arm.name, mode="offline",
                 log_dir=str(short_log_root), config=config, public=False)
    try:
        for row in lines:
            swanlab.log(row["metrics"], step=row["step"])
    except BaseException:
        swanlab.finish(state="crashed")
        raise
    swanlab.finish(state="success")
    return config, short_log_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", nargs="+", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 7))
    args = parser.parse_args()
    studies = [path.resolve() for path in args.studies]
    if len(set(studies)) != len(studies):
        parser.error("Each study must be listed once")
    queue = []
    for study in studies:
        if (study.parent != (ROOT / "artifacts").resolve()
                or not re.fullmatch(r"b-(order|robust-neural|residual)[A-Za-z0-9-]+", study.name)):
            parser.error("Expected a local B upgrade study in artifacts")
        arms = sorted(p.parent for p in study.glob("*/result.json"))
        if not arms:
            parser.error(f"No completed arms in {study.name}")
        for arm in arms:
            if arm.is_symlink():
                parser.error("Arm directory cannot be a symlink")
            config, log_root = prepare(arm)
            queue.append((arm, config, log_root))
    print(f"READY: {len(queue)} completed arms", flush=True)
    if args.prepare_only:
        return 0
    connect()  # Fail early if destination/authentication is unavailable.
    context = local()

    def upload(arm, config, log_root):
        if not hasattr(context, "api"):
            context.api = connect()
        return publish_run(ROOT, arm, config, context.api, log_root=log_root)

    index = {study: {} for study in studies}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(upload, arm, config, log_root): arm for arm, config, log_root in queue}
        for future in as_completed(jobs):
            arm = jobs[future]
            try:
                receipt = future.result()
            except Exception as exc:
                receipt = {"status": "error", "failure": type(exc).__name__, "cloud_record_verified": False}
            index[arm.parent][arm.name] = receipt
            write_json(arm.parent / "swanlab_index.json", index[arm.parent])
            print(arm.parent.name, arm.name, receipt["status"], receipt.get("cloud_url", ""), flush=True)
    return int(any(not r["cloud_record_verified"] for study in index.values() for r in study.values()))


if __name__ == "__main__":
    raise SystemExit(main())

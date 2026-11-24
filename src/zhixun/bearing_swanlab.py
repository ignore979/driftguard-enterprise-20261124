"""Publish closed B run journals to the existing project and verify the remote record."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import sleep

from zhixun.cloud import compare_config, compare_points, expected_points, normalize_config
from zhixun.data import digest, write_json

WORKSPACE = "25165030"
PROJECT = "zhixun-school_from_813_1794808892434"
PROJECT_PATH = f"{WORKSPACE}/{PROJECT}"
API_HOST = "https://api.swanlab.cn"
PROJECT_URL = f"https://swanlab.cn/@{PROJECT_PATH}"


def connect():
    import swanlab

    api = swanlab.Api(host=API_HOST)
    project = api.project(PROJECT_PATH)
    if not project.project_id or project.path.strip("/") != PROJECT_PATH:
        raise ValueError("The existing SwanLab destination project could not be verified")
    return api


def verify_remote(api, run_id, name, config, points):
    remote = api.run(f"{PROJECT_PATH}/{run_id}")
    if remote.name != name or remote.state != "FINISHED":
        raise ValueError("The cloud run name or completion state differs from the local run")
    actual = remote.profile["config"]
    if isinstance(actual, str):
        actual = json.loads(actual)
    null_fields = compare_config(config, normalize_config(actual))
    checked = compare_points(points, remote.metrics(keys=list(points), all=True))
    return {
        "verification_method": "official_swanlab_sdk_state_config_and_scalar_readback",
        "checked_config_fields": len(config),
        "checked_metric_keys": len(points),
        "checked_scalar_points": checked,
        "sdk_null_to_empty_mapping_fields": null_fields,
    }


def publish_run(root: Path, folder: Path, config: dict, api, *, log_root: Path | None = None) -> dict:
    """A completed receipt and one stable cloud ID make reissuing sync safe."""
    runs = list((log_root if log_root is not None else folder / "swanlog").glob("run-*"))
    if len(runs) != 1 or runs[0].is_symlink() or not runs[0].is_dir():
        raise ValueError("Expected exactly one completed SwanLab journal directory")
    match = re.fullmatch(r"run-\d{8}_\d{6}-([a-z0-9]{8})", runs[0].name)
    if not match:
        raise ValueError("Offline SwanLab journal has an unexpected run ID")
    run_id = match[1]
    journal = runs[0] / f"run-{run_id}.swanlab"
    if journal.is_symlink() or not journal.is_file() or not journal.stat().st_size:
        raise ValueError("Offline SwanLab journal is missing or empty")
    points = expected_points([json.loads(line) for line in (folder / "history.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()])
    if not points:
        raise ValueError("Offline SwanLab history has no scalar measurements")

    target_url = f"{PROJECT_URL}/runs/{run_id}"
    config_bytes = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    identity = {"target_url": target_url, "journal_sha256": digest(journal),
                "history_sha256": digest(folder / "history.jsonl"),
                "configuration_sha256": hashlib.sha256(config_bytes).hexdigest()}
    receipt = folder / "swanlab_publication.json"
    previous = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
    if previous and any(previous.get(key) != value for key, value in identity.items()):
        raise ValueError("The SwanLab target or offline journal changed after the first upload attempt")
    if previous.get("status") == "verified":
        return previous

    def save(status, **details):
        record = {**identity, "status": status, "updated_utc": datetime.now(UTC).isoformat(),
                  "remote_run_id": run_id, "workspace": WORKSPACE, "project": PROJECT, **details}
        staging = receipt.with_suffix(".tmp")
        write_json(staging, record)
        staging.replace(receipt)
        return record

    save("sync_started")
    try:
        completed = subprocess.run([
            sys.executable, "-B", "-X", "utf8", str(root / "run.py"), "swanlab", "sync", str(runs[0]),
            "--host", API_HOST, "--workspace", WORKSPACE, "--project", PROJECT, "--id", run_id,
        ], cwd=root, stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8",
            errors="replace", timeout=120, check=False)
    except subprocess.TimeoutExpired:
        return save("sync_failed", failure="sync_timeout", cloud_record_verified=False)
    except OSError as error:
        return save("sync_failed", failure=type(error).__name__, cloud_record_verified=False)

    output = completed.stdout + completed.stderr
    cli_urls = set(re.findall(r"https://swanlab\.cn/@[^\s/]+/[^\s/]+/runs/[a-z0-9]+", output))
    if cli_urls - {target_url}:
        return save("sync_failed", failure="unexpected_destination_url", cloud_record_verified=False)
    if completed.returncode or "Confirm sync finish failed" in output:
        return save("sync_failed", failure="swanlab_sync_error", cli_exit_code=completed.returncode,
                    cloud_record_verified=False)

    for delay in (0, 1, 3):
        sleep(delay)
        try:
            checked = verify_remote(api, run_id, folder.name, config, points)
        except Exception as error:
            # SDK errors may include request details. Record only the safe exception type.
            verification_error = type(error).__name__
        else:
            return save("verified", cloud_url=target_url, cloud_record_verified=True, **checked)
    return save("verification_pending", failure=verification_error, cloud_url=target_url,
                cloud_record_verified=False)


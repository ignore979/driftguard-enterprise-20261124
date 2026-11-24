"""Exercise the official SDK frames on a local host with real PyTRiO sampling."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from run_c_model import ROOT, evaluate, metadata

sys.path.insert(0, str(ROOT / "submission/official_sdk_1_0_2"))

import numpy as np
import pytrio
from transformers import AutoTokenizer

from contest_harness.protocol import PROTOCOL
from contest_harness.round import Round
from zhixun.contest_c_runtime import DEFAULT_C_STRATEGY, train
from zhixun.data import digest, load_dataset, verify_inputs, verify_split, write_json
from zhixun.vendor.model import Model


class LocalHost:
    """Reply to actual SDK frames; platform acknowledgement is simulated locally."""

    def __init__(self, initial, folder, validation, client, tokenizer):
        self.reply = initial
        self.folder = folder
        self.validation = validation
        self.client = client
        self.tokenizer = tokenizer
        self.history = []
        self.state = None
        self.record = None
        self.candidate_dir = None
        self.calls = []

    def readline(self, size=-1):
        if self.reply is None:
            raise RuntimeError("The local host has no pending reply")
        message, self.reply = self.reply, None
        return json.dumps(message, allow_nan=False) + "\n"

    def flush(self):
        pass

    def restore_files(self, frame, directory):
        directory.mkdir()
        descriptors = []
        for item in frame["files"]:
            filename = Path(item["path"])
            if filename.is_absolute() or ".." in filename.parts:
                raise ValueError("Unsafe path in SDK local mount")
            destination = directory / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            raw = base64.b64decode(item["data_base64"], validate=True)
            destination.write_bytes(raw)
            descriptors.append({"path": filename.as_posix(), "sha256": "sha256:" + digest(destination),
                                "size_bytes": len(raw)})
        return descriptors

    def model(self, frame):
        messages = list(frame["messages"])
        # The actual contest prompt is supplied by the host. The local host uses a clearly named fixture.
        messages[0] = {**messages[0], "content": "Local SDK rehearsal on public C data.\n\n" + messages[0]["content"]}
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=False)
        if isinstance(prompt, Mapping):
            prompt = prompt["input_ids"]
        query = pytrio.ModelInput.from_ints(tokens=[int(token) for token in prompt])
        future = self.client.sample(prompt=query, num_samples=1, sampling_params=pytrio.SamplingParams(
            max_tokens=frame["max_tokens"], temperature=0, seed=42,
        ))
        result = future.result()
        response = self.tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()
        call_id = "local-sdk-" + self.folder.name + "-" + str(frame["id"])
        self.calls.append({"local_receipt_id": call_id, "raw_model_reply": response,
                           "input_tokens": result.input_tokens, "output_tokens": result.output_tokens})
        return {"call_id": call_id, "response": {"choices": [{"message": {"content": response}}]}}

    def write(self, line):
        if self.reply is not None:
            raise RuntimeError("Outstanding local host response")
        frame = json.loads(line)
        kind = frame["type"]
        if kind == "round_complete":
            self.record = frame["record"]
            self.reply = {"type": "round_ack", "accepted": True}
        else:
            response = {"type": kind.replace("_request", "_response"), "id": frame["id"]}
            if kind == "history_request":
                response["records"] = self.history
            elif kind == "model_request":
                response.update(self.model(frame))
            elif kind == "candidate_request":
                if frame["parent_checkpoint_sha256"] != self.parent["files"][0]["sha256"]:
                    raise ValueError("Candidate changed the declared parent")
                self.candidate_dir = self.folder / "received"
                files = self.restore_files(frame, self.candidate_dir)
                model = Model.load(self.candidate_dir / "checkpoint.npz")
                error = model.predict(self.validation.x) - self.validation.y
                rmse = float(np.sqrt(np.mean(error ** 2)))
                self.candidate = {"artifact_id": "artifact-" + self.folder.name,
                                  "artifact_sha256": files[0]["sha256"], "artifact": frame["artifact"],
                                  "files": files}
                response.update(artifact_id=self.candidate["artifact_id"], summary={"metrics": {"rmse": rmse}})
            elif kind == "state_request":
                self.state = self.folder / "mounted-state"
                files = self.restore_files(frame, self.state)
                self.state_hash = "sha256:" + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
                response["state_sha256"] = self.state_hash
            else:
                raise ValueError("Unknown local SDK frame")
            self.reply = response
        return len(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sampler-run", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    require = verify_inputs("C", args.source_root)
    source = args.source_root / "data/C"
    previous = source
    parent_model = Model.load(source / "checkpoint.npz")
    training = load_dataset(source / "train.jsonl", parent_model, "C")
    validation = load_dataset(source / "validation.jsonl", parent_model, "C")
    verify_split(training, validation)
    subset, engines = metadata(source / "validation.jsonl", validation.ids)
    regimes = np.rint(validation.x[:, 0]).astype(int)
    sampler_run = json.loads(args.sampler_run.read_text(encoding="utf-8"))
    assert sampler_run["status"] == "completed"
    repository = args.tokenizer.resolve()
    os.environ["MODELSCOPE_CACHE"] = str(repository.parents[3])
    tokenizer = AutoTokenizer.from_pretrained(str(repository), local_files_only=True)
    service = pytrio.ServiceClient()
    client = service.create_sampling_client(base_model=sampler_run["base_model"],
                                            model_path=sampler_run["sampler_checkpoint_path"])
    folder = ROOT / "artifacts" / f"c-host-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:4]}"
    folder.mkdir(parents=True)
    output = folder / "C-SDK-real-Sampler"
    output.mkdir()
    print("STUDY", folder, flush=True)
    projection = folder / "projection"
    projection.mkdir()
    for name in ("train.jsonl", "validation.jsonl"):
        with (source / name).open(encoding="utf-8") as original, (projection / name).open("w", encoding="utf-8") as stripped:
            for row in map(json.loads, original):
                stripped.write(json.dumps({key: row[key] for key in ("id", "inputs", "answer")}, separators=(",", ":")) + "\n")

    hash0 = "sha256:" + digest(source / "checkpoint.npz")
    parent = {"artifact_id": "official-c-parent", "artifact_sha256": hash0,
              "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [],
                           "predict_argv": ["python3", "/app/harness.py", "predict"]},
              "files": [{"path": "checkpoint.npz", "sha256": hash0,
                         "size_bytes": (source / "checkpoint.npz").stat().st_size}]}
    reg = {"scope": "official_public_C_local_SDK_host_with_real_PyTRiO_sampling",
           "strategy": DEFAULT_C_STRATEGY, "sampler_run": args.sampler_run.parent.name,
           "train_sha256": require["files"]["train.jsonl"]["sha256"],
           "validation_sha256": require["files"]["validation.jsonl"]["sha256"],
           "parent_sha256": hash0, "runtime_sha256": digest(ROOT / "src/zhixun/contest_c_runtime.py")}
    write_json(folder / "registration.json", reg)
    rounds, points, history = [], [], []
    state, state_hash = None, ""
    for i in range(1, 11):
        run_path = output / f"r{i:02d}"
        run_path.mkdir()
        start = {"protocol": PROTOCOL, "type": "start", "run_id": folder.name, "dataset_key": "C",
                 "round_index": i, "stage_id": "local-fixed-public-C", "prompt": "Local host fixture",
                 "prompt_version": "local-rehearsal", "rounds_per_dataset": 10,
                 "parent_checkpoint_sha256": parent["files"][0]["sha256"],
                 "task_contract": {"dataset_key": "C", "task": "regression", "metric": "rmse",
                                   "metric_direction": "lower",
                                   "public_projection": {"train_path": str(projection / "train.jsonl"),
                                                         "validation_path": str(projection / "validation.jsonl"),
                                                         "row_fields": ["id", "inputs", "answer"], "source_metadata": False}},
                 "model_name": sampler_run["base_model"], "parent_artifact": parent, "artifacts": [parent],
                 "state_path": str(state) if state else None, "state_sha256": state_hash, "wall_seconds": 240,
                 "current_validation": {}, "max_output_tokens": 160}
        host = LocalHost(start, run_path, validation, client, tokenizer)
        host.history, host.parent = history[:], parent
        r = Round.start(host, host)
        r.parent.root = previous  # Substitute only the mount point in this Windows host double.
        os.environ["CONTEST_WORK"] = str(run_path / "work")
        start_clock = time.perf_counter()
        train(r)
        if not r._finished or host.record is None or len(host.calls) != 1 or host.state is None:
            raise RuntimeError("Incomplete SDK round")
        state, state_hash = host.state, host.state_hash
        audit = json.loads((state / f"round-{i:02d}.json").read_text(encoding="utf-8"))
        assert host.record["model_call_ids"] == [host.calls[0]["local_receipt_id"]]
        assert host.record["history_ids"] == [record["id"] for record in history]
        is_new = audit["select"] == "candidate"
        if is_new:
            previous, parent = host.candidate_dir, host.candidate
        before, after = audit["local_public_before"], audit["local_public_after"]
        observed = after if is_new else before
        audit.update(local_ack=True, sampler_call=host.calls[0], selected_checkpoint_sha256=digest(previous / "checkpoint.npz"),
                     wall_seconds=time.perf_counter() - start_clock)
        write_json(run_path / "audit.json", audit)
        rounds.append(audit)
        points.append({"step": i, "metrics": {
            "decision/adopted": int(is_new), "training/epochs": len(audit["losses"]),
            "selected/rmse": observed["rmse"], "selected/mae": observed["mae"],
            "candidate/rmse": after["rmse"], "selected/nasa_log_mean": observed["nasa_log_mean"],
            "selected/near_end_rmse": observed["near_end_rmse"], "round/wall_seconds": audit["wall_seconds"],
            "pytrio/input_tokens": host.calls[0]["input_tokens"], "pytrio/output_tokens": host.calls[0]["output_tokens"],
        }})
        history.append({"id": f"round-{i}", "round_index": i, "selected_artifact_id": parent["artifact_id"]})
        write_json(output / "partial.json", {"registration": reg, "rounds": rounds})
        print("ROUND", i, audit["select"], f'{observed["rmse"]:.6f}', audit["violations"], flush=True)
    model = Model.load(previous / "checkpoint.npz")
    target = output / "checkpoint.npz"
    model.save(target)
    if digest(target) != digest(previous / "checkpoint.npz"):
        raise ValueError("Saved checkpoint differs from selected artifact")
    final = evaluate(model.predict(validation.x), validation, subset, engines, regimes)
    result = {"publish_configuration": reg, "publish_metrics": points, "validation": final,
              "rounds": rounds, "selected_rounds": [r["round"] for r in rounds if r["select"] == "candidate"],
              "checkpoint_sha256": digest(target)}
    write_json(output / "result.json", result)
    print("COMPLETE", output / "result.json", result["selected_rounds"], final["overall"], flush=True)


if __name__ == "__main__":
    main()

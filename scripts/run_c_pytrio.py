"""Continue the existing PyTRiO Sampler on a disjoint C policy corpus."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytrio
from transformers import AutoTokenizer

from zhixun.bearing_swanlab import PROJECT, WORKSPACE, connect, publish_run
from zhixun.c_trio_protocol import select_messages
from zhixun.data import digest, write_json
from zhixun.tracking import Recorder


BASE_MODEL = "Qwen/Qwen3.8-27B"
SEED = 42
BATCH_SIZE = 4
LEARNING_RATE = 1e-4


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(path: Path, checksum: str, tokenizer):
    if digest(path) != checksum:
        raise ValueError("The C policy corpus split digest changed")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        messages = row["messages"]
        payload = json.loads(messages[-1]["content"])
        if messages != select_messages(payload):
            raise ValueError("A decision example does not use the registered C prompt")
        completion = row["completion"]
        action = "candidate" if payload["eligible"] and payload["host_vote"] != "reject" else "parent"
        if json.loads(completion)["select"] != action:
            raise ValueError("A label conflicts with the measured decision and host vote")
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if isinstance(prompt, Mapping):
            prompt = prompt["input_ids"]
        prompt = [int(token) for token in prompt]
        response = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
        tokens = prompt + response
        weights = np.r_[np.zeros(len(prompt), dtype=np.float32), np.ones(len(response), dtype=np.float32)]
        datum = pytrio.Datum(
            model_input=pytrio.ModelInput.from_ints(tokens=tokens[:-1]),
            loss_fn_inputs={"target_tokens": np.asarray(tokens[1:], dtype=np.int32), "weights": weights[1:]},
        )
        rows.append({"prompt": prompt, "datum": datum, "completion_tokens": len(response),
                     "context": payload, "expected_action": action})
    if not rows:
        raise ValueError("No valid rows for the policy split")
    return rows


def test_loss(training, rows):
    loss = 0.0
    tokens = sum(row["completion_tokens"] for row in rows)
    for offset in range(0, len(rows), BATCH_SIZE):
        data = [row["datum"] for row in rows[offset:offset + BATCH_SIZE]]
        result = training.forward(data, "cross_entropy").result()
        loss += float(result.metrics["loss_sum"])
    return loss / tokens


def test_actions(service, tokenizer, sampler, rows, phase, destination):
    client = service.create_sampling_client(base_model=BASE_MODEL, model_path=sampler)
    params = pytrio.SamplingParams(max_tokens=128, temperature=0, seed=SEED)
    decisions = []
    target = {"candidate": [], "parent": []}
    for offset in range(0, len(rows), BATCH_SIZE):
        futures = [(row, client.sample(prompt=pytrio.ModelInput.from_ints(tokens=row["prompt"]),
                                      num_samples=1, sampling_params=params))
                   for row in rows[offset:offset+BATCH_SIZE]]
        for row, future in futures:
            result = future.result()
            answer = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()
            try:
                parsed = json.loads(answer)
            except (TypeError, json.JSONDecodeError):
                parsed = None
            valid = (isinstance(parsed, dict) and set(parsed) == {"select", "reason"}
                     and parsed["select"] in target and isinstance(parsed["reason"], str)
                     and bool(parsed["reason"].strip()))
            expected = row["expected_action"]
            action = parsed["select"] if valid else None
            target[expected].append(action == expected)
            decisions.append({"index": len(decisions), "expected": expected, "action": action,
                              "valid": valid, "output": answer, "input_tokens": result.input_tokens,
                              "output_tokens": result.output_tokens})
        write_json(destination / f"{phase}_decisions.json", decisions)
        print(phase, f"{len(decisions)}/{len(rows)}", flush=True)
    return {"rows": len(rows), "valid": sum(row["valid"] for row in decisions),
            "agreed": sum(sum(v) for v in target.values()),
            "candidate_rows": len(target["candidate"]), "parent_rows": len(target["parent"]),
            "balanced_agreement": float(np.mean([np.mean(v) for v in target.values()])),
            "incorrect_candidate_acceptances": sum(row["expected"] == "parent" and row["action"] == "candidate"
                                                   for row in decisions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("previous_run", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--modelscope-cache", type=Path)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 4:
        parser.error("Expected one to four epochs")
    tokenizer_path = args.tokenizer.resolve()
    cache = args.modelscope_cache
    if cache is None and tokenizer_path.parent.parent.name == "Qwen--Qwen3.8-27B":
        cache = tokenizer_path.parents[3]
    if cache is not None:
        os.environ["MODELSCOPE_CACHE"] = str(cache.resolve())
    source = read_json(args.corpus / "registration.json")
    summary = read_json(args.corpus / "summary.json")
    prior = read_json(args.previous_run)
    if (source["used_public_validation_or_hidden"] is not False
            or source["official_contest_labels"] is not False
            or prior["base_model"] != BASE_MODEL or prior["status"] != "completed"):
        raise ValueError("Invalid provenance or earlier validated model identity")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    train = prepare(args.corpus / "training.jsonl", summary["training"]["sha256"], tokenizer)
    held = prepare(args.corpus / "evaluation.jsonl", summary["evaluation"]["sha256"], tokenizer)
    if {tuple(row["prompt"]) for row in train} & {tuple(row["prompt"]) for row in held}:
        raise ValueError("Echoed C prompts appear in both partitions")
    nsteps = args.epochs * ((len(train) + BATCH_SIZE - 1) // BATCH_SIZE)
    out = ROOT / "artifacts" / f"tri-c-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:6]}"
    out.mkdir()
    config = {"task": "C_harness_decision_sft", "base_model": BASE_MODEL,
              "train_corpus": args.corpus.name, "train_sha256": summary["training"]["sha256"],
              "eval_sha256": summary["evaluation"]["sha256"], "train_rows": len(train),
              "eval_rows": len(held), "planned_steps": nsteps, "epochs": args.epochs,
              "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
              "seed": SEED, "scope": source["scope"], "initialization": "previous_saved_train_weights_fresh_optimizer",
              "source_sha256": digest(Path(__file__)), "prompt_sha256": digest(ROOT / "src/zhixun/c_trio_protocol.py")}
    write_json(out / "registration.json", config)
    print("RUN", out, flush=True)
    recorder = Recorder(out, config, mode="offline", project=PROJECT, workspace=WORKSPACE)
    step = 0
    receipt = None
    try:
        service = pytrio.ServiceClient()
        if BASE_MODEL not in service.get_supported_models():
            raise ValueError("Previous contest base model is not available from PyTRiO")
        training = service.create_training_client_from_state(prior["train_checkpoint_path"])
        print("Training client restored", flush=True)
        before = test_loss(training, held)
        recorder.log(0, {"pytrio/evaluation_loss_per_token": before})
        print("Initial heldout loss:", before, flush=True)
        rng = np.random.default_rng(SEED)
        for _ in range(args.epochs):
            indices = rng.permutation(len(train)).tolist()
            for offset in range(0, len(indices), BATCH_SIZE):
                batch = [train[i] for i in indices[offset:offset+BATCH_SIZE]]
                result = training.forward_backward([row["datum"] for row in batch], "cross_entropy").result()
                training.optim_step(pytrio.AdamParams(learning_rate=LEARNING_RATE)).result()
                step += 1
                observed_tokens = sum(row["completion_tokens"] for row in batch)
                loss = float(result.metrics["loss_sum"]) / observed_tokens
                recorder.log(step, {"pytrio/loss_per_token": loss,
                                    "pytrio/supervised_tokens": observed_tokens,
                                    "pytrio/optimizer_steps": step})
                write_json(out / "progress.json", {"optimizer_steps": step})
                print(f"Step {step}/{nsteps}: {loss:.6f}", flush=True)
        after = test_loss(training, held)
        recorder.log(step + 1, {"pytrio/evaluation_loss_per_token": after})
        sampler = training.save_weights_for_sampler(name=out.name + "-sampler").result()
        state = training.save_state(name=out.name + "-train").result()
        receipt = {**config, "status": "trained", "sampler_checkpoint_path": sampler.path,
                   "train_checkpoint_path": state.path, "completed_optimizer_steps": step,
                   "evaluation_loss_before": before, "evaluation_loss_after": after}
        write_json(out / "run.json", receipt)
        print("SAMPLER", sampler.path, flush=True)
        for i, (phase, model_path) in enumerate((
            ("previous", prior["sampler_checkpoint_path"]), ("trained", sampler.path),
        ), step+2):
            measurements = test_actions(service, tokenizer, model_path, held, phase, out)
            receipt[phase] = measurements
            write_json(out / "run.json", receipt)
            recorder.log(i, {f"policy/{phase}/{k}": value for k, value in measurements.items()})
        recorder.finish()
        receipt["status"] = "completed"
        write_json(out / "run.json", receipt)
    except BaseException as exc:
        with suppress(Exception):
            recorder.finish(failed=True)
        write_json(out / "failure.json", {"error_type": type(exc).__name__, "step": step})
        raise
    published = publish_run(ROOT, out, config, connect())
    receipt["tracking"] = published
    write_json(out / "run.json", receipt)
    print("RESULT", out / "run.json", published["status"], published.get("cloud_url", ""), flush=True)
    return int(not published["cloud_record_verified"])


if __name__ == "__main__":
    raise SystemExit(main())

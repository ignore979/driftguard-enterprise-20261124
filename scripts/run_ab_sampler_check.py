"""Check the trained C Sampler on actual frozen A/B planning prompts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from run_c_pytrio import ROOT, AutoTokenizer, pytrio

from zhixun.contest_runtime import parse_plan, train
from zhixun.data import digest, verify_inputs, write_json
from zhixun.vendor.model import Model


class PlanningPromptCaptured(Exception):
    pass


def frozen_prompt(task, official_root, work):
    verify_inputs(task, official_root)
    files = official_root / "data" / task
    checkpoint = files / "checkpoint.npz"
    model = Model.load(checkpoint)

    def rows(kind):
        with (files / f"{kind}.jsonl").open(encoding="utf-8") as stream:
            yield from (json.loads(line) for line in stream if line.strip())

    def capture(messages, **options):
        prompts.extend(messages)
        raise PlanningPromptCaptured()

    prompts = []
    rnd = SimpleNamespace(
        dataset_key=task, round_index=1, max_output_tokens=256,
        task={"task": model.meta["task"]}, history=lambda: [], history_ids=[],
        parent=SimpleNamespace(checkpoint_sha256=digest(checkpoint), path=lambda _: checkpoint,
                               descriptor={"checkpoint_paths": ["checkpoint.npz"], "code_paths": []}),
        data=SimpleNamespace(training_rows=lambda: rows("train"), validation_rows=lambda: rows("validation")),
        model=capture,
    )
    os.environ["CONTEST_WORK"] = str(work)
    try:
        train(rnd)
    except PlanningPromptCaptured:
        return prompts
    raise RuntimeError("Frozen planner did not generate a request")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--official-root", type=Path, required=True)
    cli.add_argument("--tokenizer", type=Path, required=True)
    cli.add_argument("--previous", type=Path, required=True)
    cli.add_argument("--current", type=Path, required=True)
    args = cli.parse_args()
    out = ROOT / "artifacts" / f"tri-c-ab-check-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:4]}"
    arm = out / "AB-frozen-plan"
    arm.mkdir(parents=True)
    receipts = {key: json.loads(file.read_text(encoding="utf-8"))
                for key, file in (("previous", args.previous), ("trained", args.current))}
    verify_tokenizer = args.tokenizer.resolve()
    os.environ["MODELSCOPE_CACHE"] = str(verify_tokenizer.parents[3])
    tokenizer = AutoTokenizer.from_pretrained(str(verify_tokenizer), local_files_only=True)
    service = pytrio.ServiceClient()
    sample_options = pytrio.SamplingParams(max_tokens=256, temperature=0, seed=42)
    pending = []
    with tempfile.TemporaryDirectory(dir=out) as scratch:
        for task in "AB":
            messages = frozen_prompt(task, args.official_root, Path(scratch))
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=False)
            if isinstance(tokens, Mapping):
                tokens = tokens["input_ids"]
            input_tokens = pytrio.ModelInput.from_ints(tokens=[int(token) for token in tokens])
            for key, receipt in receipts.items():
                client = service.create_sampling_client(base_model=receipt["base_model"],
                                                        model_path=receipt["sampler_checkpoint_path"])
                future = client.sample(prompt=input_tokens, num_samples=1, sampling_params=sample_options)
                pending.append((task, key, future))
    outputs, points = [], {}
    for task, key, future in pending:
        result = future.result()
        response = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()
        try:
            parsed = parse_plan(response)
        except (ValueError, SyntaxError, TypeError):
            parsed = None
        outputs.append({"task": task, "sampler": key, "valid": parsed is not None, "plan": parsed,
                        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                        "response": response})
        for metric in ("input_tokens", "output_tokens", "valid"):
            points[f"{task}/{key}/{metric}"] = float(outputs[-1][metric])
    write_json(arm / "result.json", {
        "publish_configuration": {"scope": "frozen_A_B_first_round_plan_smoke_only",
                                  "previous_run": args.previous.parent.name, "trained_run": args.current.parent.name},
        "publish_metrics": [{"step": 0, "metrics": points}],
        "decisions": outputs,
    })
    print("RESULT", out, [(r["task"], r["sampler"], r["valid"], r["plan"]) for r in outputs], flush=True)


if __name__ == "__main__":
    main()

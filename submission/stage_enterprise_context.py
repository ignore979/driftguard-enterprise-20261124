"""Prepare the public enterprise build context using the pinned wheel set."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    sources = lock["image_sources"]
    for path, data in sources.items():
        if sha(ROOT / path) != data["sha256"]:
            raise ValueError("Source digest mismatch: " + path)
    wheels = {sha(file): file for file in args.wheel_dir.glob("*.whl") if file.is_file()}
    expected = {item["sha256"] for item in lock["image_dependencies"].values()}
    if missing := expected - wheels.keys():
        raise ValueError(f"Missing {len(missing)} pinned Linux wheels")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for path, data in sources.items():
        dest = out / data["context_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest)
    for path, target in [
        ("submission/Dockerfile.official", "Dockerfile"), ("NOTICE.md", "NOTICE.md"),
        ("SDK_SOURCE.json", "SDK_SOURCE.json"), ("requirements-image.txt", "wheels/requirements.txt"),
    ]:
        dest = out / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest)
    for digest in expected:
        wheel = wheels[digest]
        shutil.copyfile(wheel, out / "wheels" / wheel.name)


if __name__ == "__main__":
    main()

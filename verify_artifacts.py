#!/usr/bin/env python3
"""Fail closed unless every model artifact has its accepted size and SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--write-sums", type=Path)
    args = parser.parse_args()

    expected = json.loads(args.manifest.read_text(encoding="utf-8"))["artifacts"]
    failures: list[str] = []
    sums: list[str] = []
    for entry in expected:
        path = args.directory / entry["file"]
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        actual_bytes = path.stat().st_size
        actual_sha = digest(path)
        print(f"{entry['file']} bytes={actual_bytes} sha256={actual_sha}", flush=True)
        if actual_bytes != entry["bytes"]:
            failures.append(
                f"{entry['file']}: bytes {actual_bytes} != {entry['bytes']}"
            )
        if actual_sha != entry["sha256"]:
            failures.append(
                f"{entry['file']}: sha256 {actual_sha} != {entry['sha256']}"
            )
        sums.append(f"{actual_sha}  {entry['file']}\n")

    if failures:
        raise SystemExit("Artifact verification failed:\n" + "\n".join(failures))
    if args.write_sums:
        args.write_sums.write_text("".join(sums), encoding="utf-8")
    print(f"verified={len(expected)}/{len(expected)}", flush=True)


if __name__ == "__main__":
    main()

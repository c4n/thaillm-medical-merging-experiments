#!/usr/bin/env python3
"""Fail when a vf-eval output is incomplete or contains rollout errors."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--max-errors", type=int, default=0)
    args = parser.parse_args()

    candidates = list(args.root.rglob("results.jsonl"))
    if not candidates:
        raise SystemExit(f"FATAL: no results.jsonl under {args.root}")
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    error_count = sum(row.get("error") is not None for row in rows)
    print(f"Validated {path}: rows={len(rows)}, errors={error_count}")
    if len(rows) != args.expected:
        raise SystemExit(f"FATAL: expected {args.expected} rows, found {len(rows)}")
    if error_count > args.max_errors:
        raise SystemExit(
            f"FATAL: {error_count} rollouts contain errors "
            f"(maximum allowed: {args.max_errors})"
        )


if __name__ == "__main__":
    main()

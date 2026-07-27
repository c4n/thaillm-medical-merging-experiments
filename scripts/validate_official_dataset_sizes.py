#!/usr/bin/env python3
"""Validate and record the dataset splits used by the official replica."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from datasets import load_dataset


EXPECTED_SPLITS = {
    "ThaiLLM/med-iq": {"train": 14208, "test": 200},
    "ThaiLLM/med-tool-use": {"train": 46117, "test": 5122},
}
UPSTREAM_COMMIT = "73772633663dfe02eff558a85eacbac9f617d329"


def package_versions():
    versions = {}
    for package in (
        "datasets",
        "openai",
        "transformers",
        "verifiers",
        "vllm",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--generation-config", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--run-name")
    parser.add_argument("--requested-examples", type=int)
    parser.add_argument("--actual-examples", type=int)
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--image")
    args = parser.parse_args()

    manifest = {
        "protocol": {
            "repository": "vistec-AI/thaillm-medical-post-training",
            "upstream_commit": UPSTREAM_COMMIT,
        },
        "run": {
            "name": args.run_name,
            "stage": args.stage,
            "model": args.model,
            "tokenizer": args.tokenizer,
            "requested_examples": args.requested_examples,
            "actual_examples": args.actual_examples,
            "rollouts_per_example": args.rollouts,
            "expected_rows": (
                args.actual_examples * args.rollouts
                if args.actual_examples is not None
                and args.rollouts is not None
                else None
            ),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        },
        "runtime": {
            "image": args.image,
            "python": platform.python_version(),
            "packages": package_versions(),
        },
        "datasets": {},
    }
    if args.generation_config:
        generation_bytes = args.generation_config.read_bytes()
        manifest["generation_config"] = {
            "path": str(args.generation_config),
            "sha256": hashlib.sha256(generation_bytes).hexdigest(),
            "content": json.loads(generation_bytes),
        }

    failures = []
    for dataset_name, expected in EXPECTED_SPLITS.items():
        dataset = load_dataset(dataset_name)
        actual = {split: len(rows) for split, rows in dataset.items()}
        fingerprints = {
            split: getattr(rows, "_fingerprint", None)
            for split, rows in dataset.items()
        }
        manifest["datasets"][dataset_name] = {
            "expected_splits": expected,
            "actual_splits": actual,
            "fingerprints": fingerprints,
        }
        print(
            "{}: splits={} fingerprints={}".format(
                dataset_name,
                actual,
                fingerprints,
            )
        )
        if actual != expected:
            failures.append(
                "{}: expected {}, found {}".format(dataset_name, expected, actual)
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("Wrote {}".format(args.output))

    if failures:
        raise SystemExit(
            "FATAL: official dataset snapshot mismatch:\n  - "
            + "\n  - ".join(failures)
        )


if __name__ == "__main__":
    main()

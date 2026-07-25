#!/usr/bin/env python3
"""Populate or validate the Hugging Face cache needed for merging."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = (
    "typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview",
    "ThaiLLM/ThaiLLM-8B-SFT-IQ",
    "ThaiLLM/ThaiLLM-8B-ToolUse",
    "ThaiLLM/ThaiLLM-8B-MedApp",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Only validate cached snapshots")
    args = parser.parse_args()

    for repo_id in MODELS:
        print(f"== {'checking' if args.offline else 'downloading'} {repo_id}", flush=True)
        path = snapshot_download(
            repo_id=repo_id,
            local_files_only=args.offline,
            max_workers=8,
        )
        print(f"   {Path(path)}", flush=True)


if __name__ == "__main__":
    main()

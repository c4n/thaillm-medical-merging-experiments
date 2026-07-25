#!/usr/bin/env python3
"""Populate or validate the Hugging Face datasets cache used by evaluation."""

import argparse
import os


DATASETS = (
    "ThaiLLM/med-tool-use",
    "ThaiLLM/med-iq",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only validate that both datasets can be loaded from the cache",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Import after setting offline mode because datasets reads these variables
    # while its configuration module is initialized.
    from datasets import load_dataset

    for repo_id in DATASETS:
        print(f"== {'checking' if args.offline else 'downloading'} {repo_id}", flush=True)
        dataset = load_dataset(repo_id)
        split_sizes = {name: len(split) for name, split in dataset.items()}
        print(f"   splits: {split_sizes}", flush=True)


if __name__ == "__main__":
    main()

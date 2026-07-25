#!/usr/bin/env python3
"""Merge the published ToolUse PEFT adapter into the Typhoon base model."""


import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if (args.output / "config.json").is_file() and (args.output / "model.safetensors.index.json").is_file():
        print(f"Materialized ToolUse model already exists: {args.output}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading base model: {args.base}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    print(f"Loading ToolUse adapter: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    print("Merging LoRA weights into the base model", flush=True)
    model = model.merge_and_unload(progressbar=True)
    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    AutoTokenizer.from_pretrained(args.base, local_files_only=True).save_pretrained(args.output)
    print(f"Saved full ToolUse model to {args.output}", flush=True)


if True:
    main()

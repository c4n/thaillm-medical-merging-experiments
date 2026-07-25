#!/usr/bin/env python3
"""Render mergekit recipes for the ThaiLLM IQ/ToolUse experiment."""

import argparse
from pathlib import Path

BASE = "typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview"
IQ = "ThaiLLM/ThaiLLM-8B-SFT-IQ"
TOOL = "ThaiLLM/ThaiLLM-8B-ToolUse"
MEDAPP = "ThaiLLM/ThaiLLM-8B-MedApp"


def task_arithmetic(alpha: float) -> str:
    return f"""merge_method: task_arithmetic
base_model: {BASE}
models:
  - model: {IQ}
    parameters:
      weight: 1.0
  - model: {TOOL}
    parameters:
      weight: {alpha:.1f}
parameters:
  normalize: false
dtype: bfloat16
tokenizer:
  source: {IQ}
chat_template: auto
"""


def dare_ties(alpha: float, density: float) -> str:
    return f"""merge_method: dare_ties
base_model: {BASE}
models:
  - model: {IQ}
    parameters:
      weight: 1.0
      density: {density:.1f}
  - model: {TOOL}
    parameters:
      weight: {alpha:.1f}
      density: {density:.1f}
parameters:
  normalize: false
  int8_mask: true
dtype: bfloat16
tokenizer:
  source: {IQ}
chat_template: auto
"""


def medapp_tool_linear(tool_weight: float) -> str:
    medapp_weight = 1.0 - tool_weight
    return f"""merge_method: linear
models:
  - model: {MEDAPP}
    parameters:
      weight: {medapp_weight:.1f}
  - model: {TOOL}
    parameters:
      weight: {tool_weight:.1f}
parameters:
  normalize: true
dtype: bfloat16
tokenizer:
  source: {MEDAPP}
chat_template: auto
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("configs/merge/generated"))
    parser.add_argument("--local-models", action="store_true")
    parser.add_argument("--tool-model", type=Path)
    args = parser.parse_args()

    if args.local_models:
        from huggingface_hub import snapshot_download

        global BASE, IQ, TOOL, MEDAPP
        BASE = snapshot_download(BASE, local_files_only=True)
        IQ = snapshot_download(IQ, local_files_only=True)
        MEDAPP = snapshot_download(MEDAPP, local_files_only=True)
        if not args.tool_model:
            parser.error("--tool-model is required with --local-models because ToolUse is a LoRA adapter")
        if not (args.tool_model / "config.json").is_file():
            parser.error(f"materialized ToolUse model not found: {args.tool_model}")
        TOOL = str(args.tool_model.absolute())
        print(f"Using cached base: {BASE}")
        print(f"Using cached SFT-IQ: {IQ}")
        print(f"Using materialized ToolUse: {TOOL}")
        print(f"Using cached MedApp: {MEDAPP}")

    args.output.mkdir(parents=True, exist_ok=True)
    for alpha in (0.1, 0.2, 0.3, 0.4, 0.5):
        tag = str(alpha).replace(".", "p")
        (args.output / f"task_arithmetic_a{tag}.yml").write_text(task_arithmetic(alpha))
    for alpha in (0.2, 0.3):
        for density in (0.5, 0.7):
            atag = str(alpha).replace(".", "p")
            dtag = str(density).replace(".", "p")
            (args.output / f"dare_ties_a{atag}_d{dtag}.yml").write_text(
                dare_ties(alpha, density)
            )
    for tool_weight in (0.1, 0.2, 0.3):
        tag = str(tool_weight).replace(".", "p")
        (args.output / f"medapp_tool_linear_t{tag}.yml").write_text(
            medapp_tool_linear(tool_weight)
        )


if True:
    main()

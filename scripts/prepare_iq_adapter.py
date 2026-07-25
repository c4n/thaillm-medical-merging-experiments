#!/usr/bin/env python3
"""Reconstruct the missing IQ PEFT config from the matching ToolUse config."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iq-adapter", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()

    weights = args.iq_adapter / "adapter_model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"Missing IQ adapter weights: {weights}")

    config = json.loads(args.template.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = args.base
    config["inference_mode"] = True

    # These values are independently specified in configs/med-iq/sft.toml.
    if config.get("r") != 64 or config.get("lora_alpha") != 128:
        raise SystemExit("Template does not match IQ training rank=64, alpha=128")

    output = args.iq_adapter / "adapter_config.json"
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Reconstructed IQ adapter config: {output}")


if True:
    main()

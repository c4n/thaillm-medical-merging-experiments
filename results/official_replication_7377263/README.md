# Published-protocol evaluation snapshot

This directory contains the small, reviewable outputs from the isolated
evaluation described in
[`docs/official_evaluation_replication.md`](../../docs/official_evaluation_replication.md).
Raw Verifiers JSONL, vLLM logs, model weights, and judge-level JSONL remain on
LANTA and are intentionally excluded from Git.

The evaluation follows
`vistec-AI/thaillm-medical-post-training@73772633663dfe02eff558a85eacbac9f617d329`
with three rollouts per example:

- med-IQ: 200 examples, 600 rollouts, strict JSON scoring, and
  `deepseek/deepseek-v4-flash` response judging.
- ToolUse: 5,122 examples and 15,366 rollouts. The published command requests
  5,192 examples, which Verifiers clamps to the 5,122-row test split.

## Results

### med-IQ

| Model | Format | Citations | Response | Combined | Judge errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| MedApp 70% + ToolUse 30% | 100.00% | 67.82% | 75.83% | 1.5366 | 0 |
| MedApp | 95.67% | 62.29% | 63.17% | 1.3502 | 0 |

### ToolUse

| Model | Errors | Pass@1 accuracy | Pass@1 trigger F1 | Pass@1 macro F1 | Pass@3 accuracy | Pass@3 trigger F1 | Pass@3 macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MedApp 70% + ToolUse 30% | 0 | 99.92% | 100.00% | 99.39% | 99.94% | 100.00% | 99.41% |
| MedApp | 3 | 90.36% | 87.54% | 78.30% | 94.79% | 94.36% | 81.94% |

The three MedApp rollout errors are retained and scored as incorrect rather
than selectively regenerated.

## Interpretation

The 70/30 merge is substantially stronger than the MedApp control within this
run. The MedApp control did not reproduce the model-card ToolUse targets
(98.63% accuracy, 100% trigger F1, and 89.64% macro F1), so these artifacts
should be described as a protocol-aligned reproduction rather than an exact
model-card reproduction.

The comparison JSON contains full per-class metrics. IQ summary JSON files
contain the aggregate judge scores. Protocol manifests record model paths,
dataset sizes and fingerprints, generation configuration, package versions,
and Slurm job IDs for the retained runs.

# ThaiLLM MedApp + ToolUse remediation report

**Date:** 5 August 2026

**Selected model:** 70% ThaiLLM-8B-MedApp + 30% ThaiLLM-8B-ToolUse

## Objective

ThaiLLM-8B-MedApp provided the stronger Thai medical conversation behavior,
but its tool calling was weaker than required and it could degenerate into long
or repetitive text after several dialogue turns. ThaiLLM-8B-ToolUse was highly
accurate at tool calling and stable over multiple turns, but was not the desired
Thai conversational model. The goal was to combine these complementary
capabilities without retraining either parent model.

## Technical approach

The ToolUse adapter was first materialized as a complete BF16 checkpoint. We
then used **MergeKit 0.1.4** to perform a normalized linear full-weight merge:

$$
\theta_{merged}=0.7\,\theta_{MedApp}+0.3\,\theta_{ToolUse}
$$

The merge used `normalize: true`, BF16 output, and the MedApp tokenizer with
automatic chat-template selection. Linear interpolation was selected because
both parents have compatible architecture and base lineage, it preserves
MedApp as the dominant conversational anchor, and it provides a controlled,
interpretable way to inject ToolUse behavior. The 30% ToolUse candidate was
selected as the practical capability trade-off after a bounded weight sweep.
Merging and evaluation ran on LANTA A100 40 GB GPUs using Apptainer. Separate
Python environments isolated MergeKit's Pydantic 2.10 requirement from the
newer Pydantic required by Verifiers; models and datasets were served from the
project Hugging Face cache in offline compute jobs.

## Evaluation results

The protocol-aligned evaluation followed
`vistec-AI/thaillm-medical-post-training@73772633663dfe02eff558a85eacbac9f617d329`.
Med-IQ used 200 examples with three rollouts each and
`deepseek/deepseek-v4-flash` for response judging. ToolUse used 5,122 examples
with three rollouts each. Errors were retained and scored as failures.

| Evaluation | Original MedApp | 70/30 merge | Change |
| --- | ---: | ---: | ---: |
| Med-IQ response reward | 63.17% | **75.83%** | +12.66 points |
| Med-IQ citation reward | 62.29% | **67.82%** | +5.53 points |
| ToolUse pass@1 accuracy | 90.36% | **99.92%** | +9.56 points |
| ToolUse pass@1 trigger F1 | 87.54% | **100.00%** | +12.46 points |
| ToolUse pass@1 macro F1 | 78.30% | **99.39%** | +21.09 points |

A separate multi-turn stability diagnostic (LANTA job `6065426`, run
`20260804_223124`) ran ten fixed Thai scenarios, eight turns per conversation,
two repetitions, and two decoding profiles: 320 responses per model. Under
matched published-style decoding with a 512-token diagnostic cap, MedApp
flagged 46.88% of responses, reached the cap 61 times, and had mean repetition
0.2005. The merge flagged only 3.12%, never reached the cap, and reduced mean
repetition to 0.0063—approximately a 97% reduction. ToolUse flagged 0.62%.

With the stabilized profile (`temperature=0.4`, `top_p=0.9`,
`repetition_penalty=1.05`, `max_tokens=512`), the merge and ToolUse had zero
observed flags. MedApp improved but still flagged 33.12%, indicating that
decoding controls alone did not explain or solve the original behavior. The
results therefore support the merge as the main source of the stability gain,
with conservative decoding providing an additional operational margin.

## Decision and limitations

The 70/30 linear merge is the selected candidate because it substantially
improved medical-response, citation, tool-selection, argument, and multi-turn
stability metrics while retaining MedApp as the primary model. For interactive
chat, the stabilized decoding profile is recommended initially; tool-oriented
deployments may retain a larger output cap after separate validation.

These results demonstrate strong mitigation on controlled tests, not proof
that degeneration is impossible. The multi-turn flags are automated stability
heuristics rather than medical-correctness judgments, and flagged conversations
still require human review. The MedApp control also did not reproduce all
published model-card ToolUse targets, so the benchmark is described as
protocol-aligned rather than an exact reproduction. Before public release, the
model should undergo human multi-turn review, medical-safety and refusal tests,
and monitoring on realistic application histories. Full protocol details and
curated results are available in the
[official evaluation snapshot](../results/official_replication_7377263/README.md).

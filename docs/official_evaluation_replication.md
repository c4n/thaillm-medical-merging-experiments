# Published ThaiLLM MedApp evaluation replica

This track reproduces the evaluation semantics in
[vistec-AI/thaillm-medical-post-training](https://github.com/vistec-AI/thaillm-medical-post-training)
at commit
`73772633663dfe02eff558a85eacbac9f617d329`. It is isolated from the earlier
bounded, one-rollout experiments.

All new artifacts go under:

```text
results/official_replication_7377263/
```

Nothing in the existing `results/<old-run>/` directories is read, resumed, or
overwritten.

## Protocol

| Evaluation | Unique examples | Rollouts/example | Saved rows | Length/temperature CLI |
| --- | ---: | ---: | ---: | --- |
| med-IQ | 200 | 3 | 600 | unset |
| ToolUse | 5,122 | 3 | 15,366 | unset |

The upstream ToolUse command requests `-n 5192`, but the published test split
contains 5,122 rows. Verifiers clamps the request to the dataset length. The
replica preserves the published `-n 5192` argument and validates the actual
15,366 result rows.

The merged model and MedApp control use the published MedApp generation
configuration: sampling enabled, temperature 0.4, and EOS token IDs 151645 and
151643. The evaluation command does not override temperature or maximum output
tokens.

The IQ evaluator uses strict `json.loads` behavior for format, citation, and
response scoring. LANTA GPU nodes cannot reach OpenRouter, so generation and
judging run in two phases:

1. GPU jobs generate 600 strict-scored rollouts and defer only the response
   judge.
2. A login-node command calls `deepseek/deepseek-v4-flash` using the published
   prompt and verdict rule.

This two-phase arrangement preserves scoring semantics but is an infrastructure
adaptation. The Apptainer image uses Verifiers 0.1.13.dev8 for both tasks;
upstream ToolUse used 0.1.11. The task evaluator is the same, but this is not a
bit-for-bit software-environment reproduction.

Although the model card labels the IQ citation metric “F1,” the published
evaluator computes the fraction of predicted citation IDs that are correct
after its validity/overlap checks. The replica retains that formula.

## Sync to LANTA

From the development machine:

```bash
rsync -avL \
  --no-owner \
  --no-perms \
  --chown=:lt200394 \
  --exclude '.git/' \
  --exclude '.cache/' \
  --exclude 'models/' \
  --exclude 'results/' \
  --exclude 'logs/' \
  --exclude 'apptainer/*.sif' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  /home/can/fix_tooluse_x/ \
  cudomcha@transfer.lanta.nstda.or.th:/project/lt200394-thllmV/can/fix_tooluse_x/
```

The exclusions protect the large LANTA-side models, cache, and existing
evaluation outputs.

The existing `apptainer/thaillm-merge.sif` can be reused; the isolated
evaluator modules are loaded from the bind-mounted workspace.

## Submit the two protocol controls

On a LANTA login node:

```bash
cd /project/lt200394-thllmV/can/fix_tooluse_x

bash lanta/submit_official_eval.sh iq merged medapp
bash lanta/submit_official_eval.sh tooluse merged medapp
```

This submits:

- `medapp_tool_linear_t0p3__official_r3`
- `baseline_medapp__official_r3`

The MedApp run is the protocol control. Its
[published model-card targets](https://huggingface.co/ThaiLLM/ThaiLLM-8B-MedApp)
are:

| Metric | Published MedApp |
| --- | ---: |
| IQ response | 68.33% |
| IQ citation | 68.40% |
| ToolUse accuracy | 98.63% |
| ToolUse trigger F1 | 100.00% |
| ToolUse macro F1 | 89.64% |

The source repository does not identify whether its single ToolUse model-card
row came from the aggregate script's `pass@1` or `pass@3` block. The replica
reports both. Treat the block that reproduces the MedApp control as the
card-comparable one.

Start with these two runs. Once the MedApp control is close to the published
numbers, the cached original-model aliases can reproduce the wider comparison:

```bash
# Add the original IQ comparison rows; MedApp and the merge already exist
bash lanta/submit_official_eval.sh iq typhoon sft_iq

# Add the original ToolUse comparison rows; MedApp and the merge already exist
bash lanta/submit_official_eval.sh tooluse typhoon tooluse
```

Do not submit a second job for any alias that is already running. The
Qwen3-30B-A3B comparison from the model card is not included because it is a
separate 30B model and is not part of the cached 8B-model experiment.

## Resume a timed-out ToolUse job

The unbounded, three-rollout ToolUse run is much heavier than the earlier
1,024-token evaluation. If one alias times out, submit only that alias with
resume enabled:

```bash
RESUME_EVAL=1 \
TIME_LIMIT=12:00:00 \
bash lanta/submit_official_eval.sh tooluse merged
```

Use `medapp` instead of `merged` when resuming the control. Do not change the
model behind a run name while resuming.

## Judge IQ on the login node

After both IQ GPU jobs complete:

```bash
export OPENROUTER_API_KEY='...'
bash lanta/judge_iq_official_login.sh merged medapp
```

At most 600 judge requests are made per model; strictly invalid JSON outputs
skip the judge. The script resumes network failures without touching the
earlier `iq_judged.jsonl` files. Rerun the same command until every generated
summary reports `"judge_errors": 0`.

Read the strict IQ summaries with:

```bash
find results/official_replication_7377263 \
  -name 'iq_judged_official-*.summary.json' \
  -print -exec grep -E \
    '"n"|"format_reward"|"citations_reward"|"response_reward"|"judge_errors"' \
    {} \;
```

## Read ToolUse results

After both ToolUse outputs validate at 15,366 rows:

```bash
bash lanta/aggregate_tooluse_official.sh merged medapp
```

The comparison is written to:

```text
results/official_replication_7377263/tooluse_official_comparison.txt
results/official_replication_7377263/tooluse_official_comparison.json
```

The model-card macro F1 is the arithmetic mean of the eight expected class F1
scores, including `no_tool`. Evaluator reward averages are not the published
ToolUse accuracy.

Completed rollout errors are retained rather than selectively regenerated.
They are reported in the aggregate and scored as incorrect predictions. This
avoids survivor bias and follows the published `vf-eval` workflow, which does
not discard errored rows before aggregation.

A curated result snapshot from the completed LANTA comparison is stored in
[`results/official_replication_7377263/`](../results/official_replication_7377263/).
Raw rollout and judge-level JSONL files remain on LANTA.

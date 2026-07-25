# ThaiLLM IQ + ToolUse merge experiments on LANTA

This workspace runs a controlled task-vector sweep:

`IQ + alpha * (ToolUse - Typhoon base)`, for alpha 0.1 through 0.5.

This deliberately keeps the IQ checkpoint as the conversational anchor. The
generated directory also contains a small DARE-TIES follow-up sweep, but it is
better to score task arithmetic first.

## One-time setup on LANTA

Use project storage rather than `$HOME` for checkpoints and Hugging Face cache.
Create a Python environment containing `mergekit`, `transformers`, and their
dependencies. Compute nodes may not have internet access, so download packages
and model snapshots using the access method approved for your project before
submitting the jobs.

Set the paths for your installation:

```bash
export LANTA_ACCOUNT=lt200394
export WORK_ROOT=/project/ltXXXXXX/your-user/thaillm-merge
export MERGE_ENV=/project/ltXXXXXX/your-user/envs/mergekit
export PRIME_RL_ROOT=/project/ltXXXXXX/your-user/prime-rl
export PRIME_RL_ENV=/project/ltXXXXXX/your-user/envs/prime-rl
```

Render and inspect the recipes locally:

```bash
python scripts/render_merge_configs.py
```

## Run

From this directory on the LANTA login node:

```bash
mkdir -p logs
bash lanta/submit.sh merge
myqueue
```

After all five merge-array tasks complete:

```bash
bash lanta/submit.sh eval
myqueue
```

The evaluation array assigns one A100 to each candidate. Five candidates can
therefore occupy five GPU nodes concurrently, subject to scheduling. To reduce
concurrent allocation, submit subsets with `sbatch --array=0-1`, then `2-4`.

`med_iq` uses an external LLM judge in the published repository. Export the
judge credential/configuration required by your environment before submitting
the evaluation; never put the credential in an sbatch file.

## Selection rule

Reject candidates with tool macro-F1 below 97. Among the remaining candidates,
select the highest IQ/Thai response score and inspect per-tool argument accuracy,
especially `create_reminder` and `list_reminder`. If no task-arithmetic candidate
passes, run the generated DARE-TIES recipes or return to IQ-to-tool RL with replay
and a KL penalty.

# Apptainer setup on LANTA

## 1. Sync the source tree

Run locally:

```bash
rsync -avL \
  --exclude '__pycache__/' \
  --exclude 'models/' \
  --exclude 'results/' \
  --exclude '.cache/' \
  --exclude 'apptainer/cache/' \
  --exclude 'apptainer/tmp/' \
  --exclude 'apptainer/*.sif' \
  /home/can/fix_tooluse_x/ \
  cudomcha@transfer.lanta.nstda.or.th:/project/lt200394-thllmV/can/fix_tooluse_x/
```

## 2. Build once on LANTA

On the LANTA login node:

```bash
cd /project/lt200394-thllmV/can/fix_tooluse_x
bash apptainer/build_lanta.sh
```

The build produces `apptainer/thaillm-merge.sif` and runs its `%test` section.
It requires network access to the container registry and Python package index.

If `--fakeroot` is not enabled for the account, build the same definition on a
Linux machine where Apptainer fakeroot/root builds are available, then copy the
SIF to the path above. A SIF is read-only and does not require root to execute.

## 3. Cache the three source checkpoints

The merge reads the following Hugging Face repositories:

- `typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview`
- `ThaiLLM/ThaiLLM-8B-IQ`
- `ThaiLLM/ThaiLLM-8B-ToolUse`

By default, the jobs store them under:

```text
/project/lt200394-thllmV/can/fix_tooluse_x/.cache/huggingface
```

To use a pre-existing cache instead, export `CACHE_ROOT` when submitting.

## 4. Submit

```bash
cd /project/lt200394-thllmV/can/fix_tooluse_x
mkdir -p logs
bash lanta/submit.sh merge
myqueue
```

After every merge-array task succeeds:

```bash
bash lanta/submit.sh eval
myqueue
```

Override the image location with `SIF=/absolute/path/image.sif` if necessary.

The IQ evaluation uses an external judge. Export its required credential before
submitting the evaluation array; Slurm exports the submission environment to the
job, and the evaluator container receives the relevant variables.

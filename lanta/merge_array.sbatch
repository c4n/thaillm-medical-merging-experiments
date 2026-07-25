#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH -t 06:00:00
#SBATCH -A lt200394
#SBATCH -J thaillm-merge
#SBATCH --array=0-4
#SBATCH -o logs/%x-%A_%a.out

set -euo pipefail
umask 0002

WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR}}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/models/merged}"
CACHE_ROOT="${CACHE_ROOT:-${WORK_ROOT}/.cache/huggingface}"
SIF="${SIF:-${WORK_ROOT}/apptainer/thaillm-merge.sif}"
ALPHAS=(0p1 0p2 0p3 0p4 0p5)
ALPHA_TAG="${ALPHAS[${SLURM_ARRAY_TASK_ID}]}"
CONFIG="${WORK_ROOT}/configs/merge/generated/task_arithmetic_a${ALPHA_TAG}.yml"
OUTPUT="${MODEL_ROOT}/iq_tool_task_a${ALPHA_TAG}"

mkdir -p "${MODEL_ROOT}" "${CACHE_ROOT}" "${WORK_ROOT}/logs"
test "$(stat -c %G "${MODEL_ROOT}")" = "lt200394" || { echo "FATAL: ${MODEL_ROOT} is not owned by group lt200394; run lanta/prepare_storage.sh"; exit 1; }
export HF_HOME="${CACHE_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module load Apptainer/1.1.6
test -f "${SIF}" || { echo "FATAL: missing ${SIF}; run apptainer/build_lanta.sh first"; exit 1; }

# Offline cache preflight before GPU work.
apptainer exec --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/python \
  "${WORK_ROOT}/scripts/download_models.py" --offline

apptainer exec --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/python "${WORK_ROOT}/scripts/render_merge_configs.py" \
  --local-models --output "${WORK_ROOT}/configs/merge/generated" \
  --tool-model "${MODEL_ROOT}/tooluse-full"

srun apptainer exec --nv --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/mergekit-yaml "${CONFIG}" "${OUTPUT}" --cuda --lazy-unpickle

apptainer exec --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/python -c "from transformers import AutoConfig, AutoTokenizer; AutoConfig.from_pretrained('${OUTPUT}'); AutoTokenizer.from_pretrained('${OUTPUT}')"

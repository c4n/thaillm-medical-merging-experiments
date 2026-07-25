#!/bin/bash -l
# Convex MedApp/ToolUse interpolation: 90/10, 80/20, and 70/30.
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH -t 06:00:00
#SBATCH -A lt200394
#SBATCH -J medapp-tool-merge
#SBATCH --array=0-2
#SBATCH -o logs/%x-%A_%a.out

set -euo pipefail
umask 0002

WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR}}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/models/merged}"
CACHE_ROOT="${CACHE_ROOT:-${WORK_ROOT}/.cache/huggingface}"
SIF="${SIF:-${WORK_ROOT}/apptainer/thaillm-merge.sif}"
TOOL_WEIGHTS=(0p1 0p2 0p3)
WEIGHT_TAG="${TOOL_WEIGHTS[${SLURM_ARRAY_TASK_ID}]}"
CONFIG_DIR="${WORK_ROOT}/configs/merge/generated/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIG_DIR}/medapp_tool_linear_t${WEIGHT_TAG}.yml"
OUTPUT="${MODEL_ROOT}/medapp_tool_linear_t${WEIGHT_TAG}"

mkdir -p "${MODEL_ROOT}" "${CACHE_ROOT}" "${WORK_ROOT}/logs" "${CONFIG_DIR}"
test "$(stat -c %G "${MODEL_ROOT}")" = "lt200394" || {
  echo "FATAL: ${MODEL_ROOT} is not owned by group lt200394; run lanta/prepare_storage.sh"
  exit 1
}
export HF_HOME="${CACHE_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module load Apptainer/1.1.6
test -f "${SIF}" || { echo "FATAL: missing ${SIF}"; exit 1; }
test -s "${MODEL_ROOT}/tooluse-full/config.json" || {
  echo "FATAL: missing materialized ToolUse model"
  exit 1
}

apptainer exec --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/python \
  "${WORK_ROOT}/scripts/download_models.py" --offline

apptainer exec --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/python \
  "${WORK_ROOT}/scripts/render_merge_configs.py" \
  --local-models --output "${CONFIG_DIR}" \
  --tool-model "${MODEL_ROOT}/tooluse-full"

test -s "${CONFIG}" || { echo "FATAL: missing ${CONFIG}"; exit 1; }

srun apptainer exec --nv --env HF_HOME="${CACHE_ROOT}" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/mergekit-yaml \
  "${CONFIG}" "${OUTPUT}" --cuda --lazy-unpickle

apptainer exec --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/python -c \
  "from transformers import AutoConfig, AutoTokenizer; AutoConfig.from_pretrained('${OUTPUT}'); AutoTokenizer.from_pretrained('${OUTPUT}')"

echo "COMPLETED: ${OUTPUT}"

#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH -t 02:00:00
#SBATCH -A lt200394
#SBATCH -J tooluse-full
#SBATCH -o logs/%x-%j.out

set -euo pipefail
umask 0002

WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR}}"
CACHE_ROOT="${CACHE_ROOT:-${WORK_ROOT}/.cache/huggingface}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/models/merged}"
SIF="${SIF:-${WORK_ROOT}/apptainer/thaillm-merge.sif}"
BASE_ROOT="${CACHE_ROOT}/hub/models--typhoon-ai--typhoon-s-thaillm-8b-instruct-research-preview"
ADAPTER_ROOT="${CACHE_ROOT}/hub/models--ThaiLLM--ThaiLLM-8B-ToolUse"

BASE_REV="$(cat "${BASE_ROOT}/refs/main")"
ADAPTER_REV="$(cat "${ADAPTER_ROOT}/refs/main")"
BASE="${BASE_ROOT}/snapshots/${BASE_REV}"
ADAPTER="${ADAPTER_ROOT}/snapshots/${ADAPTER_REV}"
OUTPUT="${MODEL_ROOT}/tooluse-full"

mkdir -p "${MODEL_ROOT}" "${WORK_ROOT}/logs"
module load Apptainer/1.1.6
test -f "${SIF}" || { echo "FATAL: missing ${SIF}"; exit 1; }
test -f "${BASE}/config.json" || { echo "FATAL: missing base model"; exit 1; }
test -f "${ADAPTER}/adapter_config.json" || { echo "FATAL: missing ToolUse adapter"; exit 1; }

srun apptainer exec --nv \
  --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/python "${WORK_ROOT}/scripts/materialize_tooluse.py" \
  --base "${BASE}" --adapter "${ADAPTER}" --output "${OUTPUT}"

test -f "${OUTPUT}/config.json"
test -f "${OUTPUT}/model.safetensors.index.json"
echo "ToolUse full model ready: ${OUTPUT}"

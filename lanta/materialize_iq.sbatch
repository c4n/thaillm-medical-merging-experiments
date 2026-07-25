#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH -t 02:00:00
#SBATCH -A lt200394
#SBATCH -J iq-full
#SBATCH -o logs/%x-%j.out

set -euo pipefail
umask 0002

WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR}}"
CACHE_ROOT="${CACHE_ROOT:-${WORK_ROOT}/.cache/huggingface}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/models/merged}"
SIF="${SIF:-${WORK_ROOT}/apptainer/thaillm-merge.sif}"
BASE_ROOT="${CACHE_ROOT}/hub/models--typhoon-ai--typhoon-s-thaillm-8b-instruct-research-preview"
IQ_ROOT="${CACHE_ROOT}/hub/models--ThaiLLM--ThaiLLM-8B-IQ"
TOOL_ROOT="${CACHE_ROOT}/hub/models--ThaiLLM--ThaiLLM-8B-ToolUse"

BASE="${BASE_ROOT}/snapshots/$(cat "${BASE_ROOT}/refs/main")"
IQ="${IQ_ROOT}/snapshots/$(cat "${IQ_ROOT}/refs/main")"
TOOL="${TOOL_ROOT}/snapshots/$(cat "${TOOL_ROOT}/refs/main")"
OUTPUT="${MODEL_ROOT}/iq-full"

mkdir -p "${MODEL_ROOT}" "${WORK_ROOT}/logs"
module load Apptainer/1.1.6
test -f "${SIF}" || { echo "FATAL: missing ${SIF}"; exit 1; }
test -f "${BASE}/config.json" || { echo "FATAL: missing base model"; exit 1; }
test -f "${IQ}/adapter_model.safetensors" || { echo "FATAL: missing IQ adapter"; exit 1; }
test -f "${TOOL}/adapter_config.json" || { echo "FATAL: missing config template"; exit 1; }

apptainer exec -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/python "${WORK_ROOT}/scripts/prepare_iq_adapter.py" \
  --iq-adapter "${IQ}" --template "${TOOL}/adapter_config.json" --base "${BASE}"

srun apptainer exec --nv \
  --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/python "${WORK_ROOT}/scripts/materialize_tooluse.py" \
  --base "${BASE}" --adapter "${IQ}" --output "${OUTPUT}"

test -f "${OUTPUT}/config.json"
test -f "${OUTPUT}/model.safetensors.index.json"
echo "IQ full model ready: ${OUTPUT}"

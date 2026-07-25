#!/usr/bin/env bash
# Run this on a LANTA node with internet access, before submitting GPU jobs.
set -euo pipefail
umask 0002

WORK_ROOT="${WORK_ROOT:-/project/lt200394-thllmV/can/fix_tooluse_x}"
CACHE_ROOT="${CACHE_ROOT:-${WORK_ROOT}/.cache/huggingface}"
SIF="${SIF:-${WORK_ROOT}/apptainer/thaillm-merge.sif}"

cd "${WORK_ROOT}"
module load Apptainer/1.1.6
test -f "${SIF}" || { echo "FATAL: missing ${SIF}"; exit 1; }
mkdir -p "${CACHE_ROOT}"

echo "Hugging Face cache: ${CACHE_ROOT}"
df -h "${CACHE_ROOT}"

apptainer exec \
  --env HF_HOME="${CACHE_ROOT}" \
  --env HF_HUB_OFFLINE=0 \
  --env HF_DATASETS_OFFLINE=0 \
  --env TRANSFORMERS_OFFLINE=0 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/merge-env/bin/python \
  "${WORK_ROOT}/scripts/download_models.py"

apptainer exec \
  --env HF_HOME="${CACHE_ROOT}" \
  --env HF_HUB_OFFLINE=0 \
  --env HF_DATASETS_OFFLINE=0 \
  --env TRANSFORMERS_OFFLINE=0 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" /opt/eval-env/bin/python \
  "${WORK_ROOT}/scripts/download_eval_datasets.py"

echo "All model snapshots and evaluation datasets are cached."

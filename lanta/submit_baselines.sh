#!/bin/bash -l
# Submit identical IQ or tool-use evaluations for the unmerged baselines.
set -euo pipefail
umask 0002

STAGE="${1:?Usage: $0 iq-or-tooluse}"
case "${STAGE}" in
  iq)
    TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
    STAGE_EXPORTS="EVAL_STAGE=iq,IQ_N=200,ROLLOUTS=1,MED_IQ_JUDGE_MODE=deferred"
    ;;
  tooluse)
    TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
    STAGE_EXPORTS="EVAL_STAGE=tooluse,TOOLUSE_N=5122,ROLLOUTS=1"
    ;;
  *)
    echo "Usage: $0 {iq|tooluse}" >&2
    exit 2
    ;;
esac

WORK_ROOT="${WORK_ROOT:-/project/lt200394-thllmV/can/fix_tooluse_x}"
LANTA_ACCOUNT="${LANTA_ACCOUNT:-lt200394}"
CACHE_HUB="${WORK_ROOT}/.cache/huggingface/hub"
RUN_SUFFIX="${RUN_SUFFIX:-}"
TOOLUSE_MAX_TOKENS="${TOOLUSE_MAX_TOKENS:-1024}"

snapshot_path() {
  local repo_slug="$1"
  local root="${CACHE_HUB}/models--${repo_slug//\//--}"
  local revision
  test -f "${root}/refs/main" || {
    echo "FATAL: ${repo_slug} is not cached; run apptainer/download_models_lanta.sh" >&2
    return 1
  }
  revision=$(cat "${root}/refs/main")
  realpath "${root}/snapshots/${revision}"
}

BASE_MODEL=$(snapshot_path "typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview")
IQ_MODEL=$(snapshot_path "ThaiLLM/ThaiLLM-8B-SFT-IQ")
MEDAPP_MODEL=$(snapshot_path "ThaiLLM/ThaiLLM-8B-MedApp")
TOOLUSE_MODEL="${WORK_ROOT}/models/merged/tooluse-full"
test -s "${TOOLUSE_MODEL}/config.json" || {
  echo "FATAL: materialized ToolUse model is missing: ${TOOLUSE_MODEL}" >&2
  exit 1
}

NAMES=(baseline_typhoon baseline_sft_iq baseline_tooluse baseline_medapp)
MODELS=("${BASE_MODEL}" "${IQ_MODEL}" "${TOOLUSE_MODEL}" "${MEDAPP_MODEL}")

cd "${WORK_ROOT}"
for index in "${!NAMES[@]}"; do
  name="${NAMES[${index}]}"
  model="${MODELS[${index}]}"
  echo "== submitting ${STAGE}: ${name}"
  echo "   ${model}"
  sbatch \
    -A "${LANTA_ACCOUNT}" \
    --array=0-0 \
    --time="${TIME_LIMIT}" \
    --export="MODEL_OVERRIDE=${model},RUN_NAME_OVERRIDE=${name},RUN_SUFFIX=${RUN_SUFFIX},TOOLUSE_MAX_TOKENS=${TOOLUSE_MAX_TOKENS},${STAGE_EXPORTS}" \
    lanta/eval_array.sbatch
done

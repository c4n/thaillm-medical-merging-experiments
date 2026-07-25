#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH -t 12:00:00
#SBATCH -A lt200394
#SBATCH -J thaillm-eval
#SBATCH --array=0-4
#SBATCH -o logs/%x-%A_%a.out

set -euo pipefail
umask 0002

WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR}}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/models/merged}"
CACHE_ROOT="${CACHE_ROOT:-${WORK_ROOT}/.cache/huggingface}"
RESULT_ROOT="${RESULT_ROOT:-${WORK_ROOT}/results}"
SIF="${SIF:-${WORK_ROOT}/apptainer/thaillm-merge.sif}"
ALPHAS=(0p1 0p2 0p3 0p4 0p5)
ALPHA_TAG="${ALPHAS[${SLURM_ARRAY_TASK_ID}]}"
MODEL="${MODEL_OVERRIDE:-${MODEL_ROOT}/iq_tool_task_a${ALPHA_TAG}}"
TOKENIZER="${TOKENIZER_OVERRIDE:-${MODEL}}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="${RUN_NAME_OVERRIDE:-task_a${ALPHA_TAG}}${RUN_SUFFIX}"
EVAL_STAGE="${EVAL_STAGE:-all}"
TOOLUSE_N="${TOOLUSE_N:-5192}"
IQ_N="${IQ_N:-200}"
ROLLOUTS="${ROLLOUTS:-3}"
TOOLUSE_MAX_TOKENS="${TOOLUSE_MAX_TOKENS:-1024}"
TOOLUSE_MAX_ERRORS="${TOOLUSE_MAX_ERRORS:-5}"
IQ_MAX_ERRORS="${IQ_MAX_ERRORS:-0}"
EVAL_PYTHONPATH="${WORK_ROOT}/environments/med_iq:${WORK_ROOT}/environments/med_tooluse"
MED_IQ_JUDGE_MODE="${MED_IQ_JUDGE_MODE:-deferred}"
RESUME_EVAL="${RESUME_EVAL:-0}"
VF_RESUME_ARGS=()
if [ "${RESUME_EVAL}" = 1 ]; then
  VF_RESUME_ARGS=(--resume)
fi
JOB_PORT_SEED="${SLURM_JOB_ID:-$$}"
TASK_PORT_SEED="${SLURM_ARRAY_TASK_ID:-0}"
PORT=$((20000 + (JOB_PORT_SEED + TASK_PORT_SEED) % 20000))

case "${EVAL_STAGE}" in
  all|tooluse|iq) ;;
  *) echo "FATAL: EVAL_STAGE must be all, tooluse, or iq" >&2; exit 2 ;;
esac
if [ "${EVAL_STAGE}" != tooluse ] && [ "${MED_IQ_JUDGE_MODE}" = online ]; then
  : "${OPENROUTER_API_KEY:?Export OPENROUTER_API_KEY before submitting an IQ evaluation job}"
fi

mkdir -p "${CACHE_ROOT}" "${RESULT_ROOT}/${RUN_NAME}" "${WORK_ROOT}/logs"
export HF_HOME="${CACHE_ROOT}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
module load Apptainer/1.1.6
test -f "${SIF}" || { echo "FATAL: missing ${SIF}; run apptainer/build_lanta.sh first"; exit 1; }
cd "${WORK_ROOT}"

echo "== model: ${MODEL}"
echo "== tokenizer: ${TOKENIZER}"
echo "== run: ${RUN_NAME}; stage: ${EVAL_STAGE}; rollouts: ${ROLLOUTS}"
echo "== samples: tooluse=${TOOLUSE_N}; iq=${IQ_N}"
echo "== tool-use max tokens: ${TOOLUSE_MAX_TOKENS}"
echo "== med-IQ judge mode: ${MED_IQ_JUDGE_MODE}"
echo "== resume evaluation: ${RESUME_EVAL}"
echo "== vLLM port: ${PORT}"

echo "== validating cached evaluation datasets"
apptainer exec --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project "${SIF}" \
  /opt/eval-env/bin/python "${WORK_ROOT}/scripts/download_eval_datasets.py" --offline

apptainer exec --nv --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project \
  "${SIF}" vllm serve "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --served-model-name "${RUN_NAME}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port "${PORT}" \
  >"${RESULT_ROOT}/${RUN_NAME}/vllm-${EVAL_STAGE}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

apptainer exec -B "${WORK_ROOT}:${WORK_ROOT}" "${SIF}" \
  python3 - "${PORT}" "${RUN_NAME}" <<'PY'
import json, sys, time, urllib.request
url = f"http://127.0.0.1:{sys.argv[1]}/health"
for _ in range(180):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("vLLM did not become healthy within six minutes")

models_url = f"http://127.0.0.1:{sys.argv[1]}/v1/models"
with urllib.request.urlopen(models_url, timeout=5) as response:
    model_ids = {item["id"] for item in json.load(response).get("data", [])}
expected = sys.argv[2]
if expected not in model_ids:
    raise SystemExit(f"Wrong vLLM server on port {sys.argv[1]}: expected {expected!r}, found {sorted(model_ids)!r}")
PY

export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=local

if [ "${EVAL_STAGE}" = all ] || [ "${EVAL_STAGE}" = tooluse ]; then
  apptainer exec --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env OPENAI_BASE_URL="${OPENAI_BASE_URL}" --env OPENAI_API_KEY=local \
    --env PYTHONPATH="${EVAL_PYTHONPATH}" \
    -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project "${SIF}" \
    /opt/eval-env/bin/vf-eval med_app_env --model "${RUN_NAME}" \
    --api-base-url "${OPENAI_BASE_URL}" --api-key-var OPENAI_API_KEY \
    -n "${TOOLUSE_N}" -r "${ROLLOUTS}" --max-tokens "${TOOLUSE_MAX_TOKENS}" \
    "${VF_RESUME_ARGS[@]}" \
    -o "${RESULT_ROOT}/${RUN_NAME}/vf" --save-results 2>&1 | \
    tee "${RESULT_ROOT}/${RUN_NAME}/tooluse.txt"
  apptainer exec -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project "${SIF}" \
    /opt/eval-env/bin/python "${WORK_ROOT}/scripts/validate_eval_results.py" \
    --root "${RESULT_ROOT}/${RUN_NAME}/vf/evals/med_app_env--${RUN_NAME}" \
    --expected "$((TOOLUSE_N * ROLLOUTS))" \
    --max-errors "${TOOLUSE_MAX_ERRORS}"
fi

if [ "${EVAL_STAGE}" = all ] || [ "${EVAL_STAGE}" = iq ]; then
  apptainer exec --env HF_HOME="${CACHE_ROOT}" --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env OPENAI_BASE_URL="${OPENAI_BASE_URL}" --env OPENAI_API_KEY=local \
    --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
    --env MED_IQ_DEBUG="${MED_IQ_DEBUG:-0}" \
    --env MED_IQ_JUDGE_MODE="${MED_IQ_JUDGE_MODE}" \
    --env PYTHONPATH="${EVAL_PYTHONPATH}" \
    -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project "${SIF}" \
    /opt/eval-env/bin/vf-eval med_iq --model "${RUN_NAME}" \
    --api-base-url "${OPENAI_BASE_URL}" --api-key-var OPENAI_API_KEY \
    -n "${IQ_N}" -r "${ROLLOUTS}" -C judge_response \
    "${VF_RESUME_ARGS[@]}" \
    -o "${RESULT_ROOT}/${RUN_NAME}/vf" --save-results 2>&1 | \
    tee "${RESULT_ROOT}/${RUN_NAME}/iq.txt"
  apptainer exec -B "${WORK_ROOT}:${WORK_ROOT}" -B /project:/project "${SIF}" \
    /opt/eval-env/bin/python "${WORK_ROOT}/scripts/validate_eval_results.py" \
    --root "${RESULT_ROOT}/${RUN_NAME}/vf/evals/med_iq--${RUN_NAME}" \
    --expected "$((IQ_N * ROLLOUTS))" \
    --max-errors "${IQ_MAX_ERRORS}"
fi

#!/bin/bash
set -euo pipefail
umask 0002

LANTA_ACCOUNT="${LANTA_ACCOUNT:-lt200394}"

MODE="${1:-merge}"
MATERIALIZE_JOB=""
case "${MODE}" in
  materialize)
    sbatch -A "${LANTA_ACCOUNT}" lanta/materialize_tooluse.sbatch
    ;;
  merge)
    sbatch -A "${LANTA_ACCOUNT}" lanta/merge_array.sbatch
    ;;
  eval)
    sbatch -A "${LANTA_ACCOUNT}" lanta/eval_array.sbatch
    ;;
  *)
    echo "Usage: $0 {materialize|merge|eval}" >&2
    exit 2
    ;;
esac

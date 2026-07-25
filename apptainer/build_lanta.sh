#!/usr/bin/env bash
set -euo pipefail
umask 0002

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load Apptainer/1.1.6

mkdir -p apptainer/cache apptainer/tmp
export APPTAINER_CACHEDIR="${PWD}/apptainer/cache"
export APPTAINER_TMPDIR="${PWD}/apptainer/tmp"

apptainer build --fakeroot \
    apptainer/thaillm-merge.sif \
    apptainer/thaillm-merge.def

apptainer test apptainer/thaillm-merge.sif

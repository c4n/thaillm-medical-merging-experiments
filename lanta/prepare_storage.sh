#!/usr/bin/env bash
# Repair LANTA project group ownership and preserve it for future files.
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/project/lt200394-thllmV/can/fix_tooluse_x}"
PROJECT_GROUP="${PROJECT_GROUP:-lt200394}"

test -d "${WORK_ROOT}" || { echo "FATAL: missing ${WORK_ROOT}"; exit 1; }
cd "${WORK_ROOT}"

echo "Before repair:"
ls -ld "${WORK_ROOT}" models models/merged .cache .cache/huggingface 2>/dev/null || true

echo "Changing group ownership to ${PROJECT_GROUP} (this can take a while)..."
chgrp -R "${PROJECT_GROUP}" "${WORK_ROOT}"

echo "Applying SetGID to every existing directory..."
find "${WORK_ROOT}" -type d -exec chmod g+s {} +

echo "Directories/files still outside ${PROJECT_GROUP}:"
find "${WORK_ROOT}" ! -group "${PROJECT_GROUP}" -print | head -20

echo "After repair:"
ls -ld "${WORK_ROOT}" models models/merged .cache .cache/huggingface 2>/dev/null || true
echo "Storage group repair complete. New files will inherit ${PROJECT_GROUP}."

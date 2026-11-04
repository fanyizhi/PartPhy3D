#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/run_abo500_mesh.sh ABO500_ROOT MESH_ROOT MESH_UNIT_TO_METER [OUTPUT]"
  exit 2
fi

DATA_ROOT="$1"
MESH_ROOT="$2"
MESH_UNIT_TO_METER="$3"
OUTPUT_ROOT="${4:-outputs/abo500_mesh}"
CONFIG="configs/abo500_mesh.yaml"

python -m tactile3d.cli abo500-validate \
  --root "${DATA_ROOT}" \
  --mesh-root "${MESH_ROOT}" \
  --mesh-unit-to-meter "${MESH_UNIT_TO_METER}"

python -m tactile3d.cli abo500-run \
  --root "${DATA_ROOT}" \
  --mesh-root "${MESH_ROOT}" \
  --mesh-unit-to-meter "${MESH_UNIT_TO_METER}" \
  --config "${CONFIG}" \
  --split train \
  --output "${OUTPUT_ROOT}/train"

python -m tactile3d.cli abo500-calibrate \
  --features "${OUTPUT_ROOT}/train/features.jsonl" \
  --config "${CONFIG}" \
  --output "${OUTPUT_ROOT}/mass_calibration.json"

for SPLIT in val test; do
  python -m tactile3d.cli abo500-run \
    --root "${DATA_ROOT}" \
    --mesh-root "${MESH_ROOT}" \
    --mesh-unit-to-meter "${MESH_UNIT_TO_METER}" \
    --config "${CONFIG}" \
    --split "${SPLIT}" \
    --calibration "${OUTPUT_ROOT}/mass_calibration.json" \
    --output "${OUTPUT_ROOT}/${SPLIT}"

  python -m tactile3d.cli abo500-evaluate \
    --predictions "${OUTPUT_ROOT}/${SPLIT}/predictions.json" \
    --output "${OUTPUT_ROOT}/${SPLIT}/metrics.json"
done

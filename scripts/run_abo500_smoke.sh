#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs/abo500_smoke/data/abo_500}"
MESH_ROOT="${ROOT}_meshes"
OUTPUT="${2:-outputs/abo500_smoke/run}"
CONFIG="configs/abo500_mesh_smoke.yaml"

python -m tactile3d.cli abo500-make-demo --root "${ROOT}"
python -m tactile3d.cli abo500-run \
  --root "${ROOT}" --mesh-root "${MESH_ROOT}" --mesh-unit-to-meter 1 \
  --config "${CONFIG}" --split train --output "${OUTPUT}/train"
python -m tactile3d.cli abo500-calibrate \
  --features "${OUTPUT}/train/features.jsonl" --config "${CONFIG}" \
  --output "${OUTPUT}/mass_calibration.json"
python -m tactile3d.cli abo500-run \
  --root "${ROOT}" --mesh-root "${MESH_ROOT}" --mesh-unit-to-meter 1 \
  --config "${CONFIG}" --split test --calibration "${OUTPUT}/mass_calibration.json" \
  --output "${OUTPUT}/test"
python -m tactile3d.cli abo500-evaluate \
  --predictions "${OUTPUT}/test/predictions.json" \
  --output "${OUTPUT}/test/metrics.json"

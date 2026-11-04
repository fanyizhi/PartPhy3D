#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT_ROOT="${TACTILE3D_TEST_OUTPUT:-outputs/server_smoke}"
SYNTHETIC_ROOT="${OUTPUT_ROOT}/synthetic/data/abo_500"
SYNTHETIC_RUN="${OUTPUT_ROOT}/synthetic/run"

echo "[1/7] Environment"
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi is unavailable; CPU checks can continue."
fi
if command -v blender >/dev/null 2>&1; then
  blender --version | head -n 1
else
  echo "INFO: Blender is unavailable; mesh-only internal rendering is not tested."
fi

echo "[2/7] Unit and integration tests"
python -m pytest -q

echo "[3/7] Synthetic ABO-shaped end-to-end run"
bash scripts/run_abo500_smoke.sh "${SYNTHETIC_ROOT}" "${SYNTHETIC_RUN}"

echo "[4/7] Watertight volume integration"
python -m tactile3d.cli mesh-volume \
  --mesh "${SYNTHETIC_RUN}/test/scenes/scene0004/mesh_metric.ply" \
  --mesh-unit-to-meter 1 \
  --repair-strategy fill_only \
  --density 650 \
  --fill-fraction 0.75 \
  --output "${OUTPUT_ROOT}/volume_report.json"

echo "[5/7] Part replacement and rigid mass-property reconstruction"
python -m tactile3d.cli part-replace \
  --source "${SYNTHETIC_RUN}/test/scenes/scene0004/mesh_metric.ply" \
  --face-labels "${SYNTHETIC_RUN}/test/scenes/scene0004/face_labels.npy" \
  --target-part 1 \
  --donor "${SYNTHETIC_RUN}/train/scenes/scene0000/mesh_metric.ply" \
  --mesh-unit-to-meter 1 \
  --source-density 650 \
  --donor-density 2700 \
  --no-icp \
  --output "${OUTPUT_ROOT}/part_edit"

echo "[6/7] Verify required reports"
python - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = [
    root / "synthetic/run/test/metrics.json",
    root / "volume_report.json",
    root / "part_edit/edit_report.json",
    root / "part_edit/edited_assembly.ply",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"Missing server-smoke outputs: {missing}")
report = json.loads((root / "part_edit/edit_report.json").read_text(encoding="utf-8"))
checks = report["physics"]["rigid_body_consistency"]
if not checks["all_passed"]:
    raise SystemExit(f"Rigid-body checks failed: {checks}")
print("Synthetic metrics:", (root / "synthetic/run/test/metrics.json").resolve())
print("Volume report:", (root / "volume_report.json").resolve())
print("Edit report:", (root / "part_edit/edit_report.json").resolve())
PY

echo "[7/7] Optional real ABO-500 two-object check"
if [[ $# -eq 0 ]]; then
  echo "SKIPPED: pass ABO_ROOT MESH_ROOT MESH_UNIT_TO_METER to enable it."
elif [[ $# -eq 3 ]]; then
  ABO_ROOT="$1"
  MESH_ROOT="$2"
  MESH_UNIT_TO_METER="$3"
  python -m tactile3d.cli abo500-validate \
    --root "${ABO_ROOT}" \
    --mesh-root "${MESH_ROOT}" \
    --mesh-unit-to-meter "${MESH_UNIT_TO_METER}" \
    --split test \
    --limit 2
  python -m tactile3d.cli abo500-run \
    --root "${ABO_ROOT}" \
    --mesh-root "${MESH_ROOT}" \
    --mesh-unit-to-meter "${MESH_UNIT_TO_METER}" \
    --config configs/abo500_mesh.yaml \
    --split test \
    --limit 2 \
    --output "${OUTPUT_ROOT}/abo500_real_two"
  python -m tactile3d.cli abo500-evaluate \
    --predictions "${OUTPUT_ROOT}/abo500_real_two/predictions.json" \
    --output "${OUTPUT_ROOT}/abo500_real_two/metrics.json"
else
  echo "Usage: bash scripts/server_smoke_test.sh [ABO_ROOT MESH_ROOT MESH_UNIT_TO_METER]" >&2
  exit 2
fi

echo "PASS: PartPhy3D server smoke test completed. Outputs: ${OUTPUT_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

MESH=${1:?usage: run_mesh_only.sh MESH UNIT_TO_METER OUTPUT [BLENDER]}
UNIT_TO_METER=${2:?usage: run_mesh_only.sh MESH UNIT_TO_METER OUTPUT [BLENDER]}
OUTPUT=${3:?usage: run_mesh_only.sh MESH UNIT_TO_METER OUTPUT [BLENDER]}
BLENDER=${4:-blender}

tactile3d mesh-run \
  --mesh "$MESH" \
  --mesh-unit-to-meter "$UNIT_TO_METER" \
  --output "$OUTPUT" \
  --blender "$BLENDER" \
  --config configs/abo500_mesh_vlm.yaml

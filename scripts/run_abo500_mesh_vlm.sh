#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 ABO_ROOT MESH_ROOT MESH_UNIT_TO_METER OUTPUT_DIR [LIMIT]" >&2
  exit 2
fi

ABO_ROOT="$1"
MESH_ROOT="$2"
MESH_UNIT_TO_METER="$3"
OUTPUT_DIR="$4"
LIMIT="${5:-}"
CONFIG="configs/abo500_mesh_vlm.yaml"
LIMIT_ARGS=()
if [ -n "$LIMIT" ]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

tactile3d abo500-validate \
  --root "$ABO_ROOT" \
  --mesh-root "$MESH_ROOT" \
  --mesh-unit-to-meter "$MESH_UNIT_TO_METER" \
  --split all \
  "${LIMIT_ARGS[@]}"

tactile3d abo500-run \
  --root "$ABO_ROOT" \
  --mesh-root "$MESH_ROOT" \
  --mesh-unit-to-meter "$MESH_UNIT_TO_METER" \
  --config "$CONFIG" \
  --split train \
  --output "$OUTPUT_DIR/train" \
  "${LIMIT_ARGS[@]}"

tactile3d abo500-calibrate \
  --features "$OUTPUT_DIR/train/features.jsonl" \
  --config "$CONFIG" \
  --output "$OUTPUT_DIR/mass_calibration.json"

tactile3d abo500-run \
  --root "$ABO_ROOT" \
  --mesh-root "$MESH_ROOT" \
  --mesh-unit-to-meter "$MESH_UNIT_TO_METER" \
  --config "$CONFIG" \
  --split test \
  --calibration "$OUTPUT_DIR/mass_calibration.json" \
  --output "$OUTPUT_DIR/test" \
  "${LIMIT_ARGS[@]}"

tactile3d abo500-evaluate \
  --predictions "$OUTPUT_DIR/test/predictions.json" \
  --output "$OUTPUT_DIR/test/metrics.json"

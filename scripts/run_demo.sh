#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/demo
python -m tactile3d.cli make-demo-image --output outputs/demo/demo.png
tactile3d infer --image outputs/demo/demo.png --config configs/default.yaml --output outputs/demo
cat outputs/demo/summary.json

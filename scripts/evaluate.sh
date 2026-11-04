#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/evaluate.sh outputs/demo/summary.json data/target.json"
  exit 1
fi

tactile3d evaluate --pred "$1" --target "$2"

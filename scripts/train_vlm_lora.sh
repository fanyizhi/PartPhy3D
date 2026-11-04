#!/usr/bin/env bash
set -euo pipefail

DATA=${1:?usage: train_vlm_lora.sh PREPARED_DATA OUTPUT [MODEL]}
OUTPUT=${2:?usage: train_vlm_lora.sh PREPARED_DATA OUTPUT [MODEL]}
MODEL=${3:-Qwen/Qwen3-VL-8B-Instruct}

tactile3d train-vlm \
  --model "$MODEL" \
  --train-data "$DATA/train.jsonl" \
  --val-data "$DATA/val.jsonl" \
  --output "$OUTPUT" \
  --epochs 3 \
  --learning-rate 2e-4 \
  --gradient-accumulation 16

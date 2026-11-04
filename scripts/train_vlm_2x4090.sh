#!/usr/bin/env bash
set -euo pipefail

DATA=${1:?usage: train_vlm_2x4090.sh PREPARED_DATA OUTPUT [MODEL]}
OUTPUT=${2:?usage: train_vlm_2x4090.sh PREPARED_DATA OUTPUT [MODEL]}
MODEL=${3:-Qwen/Qwen3-VL-8B-Instruct}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nproc_per_node=2 -m tactile3d.cli train-vlm \
  --model "$MODEL" \
  --train-data "$DATA/train.jsonl" \
  --val-data "$DATA/val.jsonl" \
  --output "$OUTPUT" \
  --qlora \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --max-pixels $((384 * 28 * 28)) \
  --epochs 3 \
  --learning-rate 2e-4

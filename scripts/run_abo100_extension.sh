#!/usr/bin/env bash
set -euo pipefail

base=${1:?experiment directory}
py=${2:?Python interpreter}
models=${3:?model directory}
prior=${4:?archived artifacts directory}
weights=${5:?evaluation weights}
nerf=${6:?official baseline directory}
mode=${7:-full}

run() { "$py" "scripts/$1.py" "${@:2}"; }
wait_for_reconstruction() {
  # Failed meshes have explicit exit codes and will be counted as fallbacks.
  until "$py" -c 'import json,sys; assert len(json.load(open(sys.argv[1]))["rows"]) >= int(sys.argv[2])' \
    "$base/reconstructions_sparse/progress.json" "$1" 2>/dev/null; do
    if ! pgrep -f "run_abo50_reconstruction.py.*--output $base/reconstructions_sparse$" >/dev/null; then
      echo 'Reconstruction exited before writing the required statuses' >&2
      return 1
    fi
    sleep 15
  done
}

full_base=$base
model="$models/Qwen3-VL-8B-Instruct"
prepare_batch() {
repair_base="$base/repaired_pipeline"
mkdir "$repair_base"
ln -s "$base/observations" "$repair_base/observations"
ln -s "$full_base/reconstructions_sparse" "$repair_base/reconstructions"
run run_abo100_extension prepare-clip --observations "$base/observations" --output "$base/clip_inputs"
run run_abo_image_mass_control predict --dataset "$base/clip_inputs" \
  --checkpoint "$models/open_clip_pytorch_model.bin" --frozen-head "$prior/mass_kg_head.npz" --output "$base/clip"
run run_abo50_mass_hypotheses --observations "$base/observations" --reconstructions "$repair_base/reconstructions" \
  --model "$model" --output "$repair_base/mass_hypotheses"
run replay_abo50_mass_hypotheses --image-only-run "$repair_base/mass_hypotheses" \
  --repair-winding --reconstructions "$repair_base/reconstructions" --output "$repair_base/winding_only"
}

finish_batch() {
repair_base="$base/repaired_pipeline"
run repair_abo50_construction --benchmark "$repair_base" --model "$model" --output "$repair_base/construction_retry"
run repair_abo50_construction --benchmark "$repair_base" --model "$model" --weighted-followup --output "$base/weighted"
run replay_abo50_open_frames --benchmark "$repair_base" --source "$base/weighted" --model "$model" \
  --output "$base/canopies" --keep-valid-source-on-failure
run replay_abo50_open_frames --benchmark "$repair_base" --source "$base/canopies" --model "$model" \
  --output "$base/panels" --panel-enclosures --keep-valid-source-on-failure
select_args=()
run run_abo100_extension appliance-candidates --construction "$base/panels" >"$base/appliance_candidates.txt"
mapfile -t candidates < <(sed '/^$/d' "$base/appliance_candidates.txt")
if ((${#candidates[@]})); then
  run run_appliance_mass_probe --observations "$base/observations" --model "$model" \
    --output "$base/appliance_raw" --scenes "${candidates[@]}"
  run replay_appliance_role_constraints --run "$base/appliance_raw" --output "$base/appliance" --skip-invalid
  select_args+=(--appliance "$base/appliance")
fi
run run_abo100_extension select --cohort "$base/cohort75.json" --previous "$prior/selected_ours" \
  --clip "$base/clip" --construction "$base/panels" --output "$base/selected" "${select_args[@]}"
}

if [[ "$mode" == standalone-batch ]]; then
  # Standalone batches already have verified reconstructions and a selection manifest.
  prepare_batch
  finish_batch
  exit 0
elif [[ "$mode" == full ]]; then
  run run_abo100_extension partition --cohort "$full_base/cohort100.json" \
    --observations "$full_base/observations" --output "$full_base"
  wait_for_reconstruction 25
  base="$full_base/batch_00"
  prepare_batch
  finish_batch
  run run_abo100_extension score --cohort "$base/cohort75.json" --run "$base/selected" \
    --weights "$weights" --nerf "$nerf" --output "$full_base/score75.json"
  base=$full_base
  wait_for_reconstruction 50
  base="$full_base/batch_25"
  prepare_batch
elif [[ "$mode" == resume-after-winding ]]; then
  base="$full_base/batch_25"
  # All three completed inputs must match their sealed bytes before reusing them.
  "$py" -c 'import sys, hashlib, json; from pathlib import Path
for directory in sys.argv[1:]:
    p = Path(directory)
    actual = hashlib.sha256((p / "predictions.json").read_bytes()).hexdigest()
    if actual != json.loads((p / "prediction_seal.json").read_text())["sha256"]:
        raise ValueError(f"Prediction seal mismatch: {p}")' \
    "$base/clip" "$base/repaired_pipeline/mass_hypotheses" "$base/repaired_pipeline/winding_only"
else
  echo "Unknown run mode: $mode" >&2
  exit 2
fi
finish_batch
run run_abo100_extension merge --cohort "$full_base/cohort100.json" \
  --first "$full_base/batch_00/selected" --second "$full_base/batch_25/selected" --output "$full_base/selected"
run run_abo100_extension score --cohort "$full_base/cohort100.json" --run "$full_base/selected" \
  --weights "$weights" --nerf "$nerf" --output "$full_base/score.json"

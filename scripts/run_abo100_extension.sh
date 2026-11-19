#!/usr/bin/env bash
set -euo pipefail

base=${1:?experiment directory}
py=${2:?Python interpreter}
models=${3:?model directory}
prior=${4:?archived artifacts directory}
weights=${5:?evaluation weights}
nerf=${6:?official baseline directory}

run() { "$py" "scripts/$1.py" "${@:2}"; }
wait_for_reconstruction() {
  # Failed meshes have explicit exit codes and will be counted as fallbacks.
  until "$py" -c 'import json,sys; assert len(json.load(open(sys.argv[1]))["rows"]) == 50' \
    "$base/reconstructions_sparse/progress.json" 2>/dev/null; do
    if ! pgrep -f "run_abo50_reconstruction.py.*--output $base/reconstructions_sparse$" >/dev/null; then
      echo 'Reconstruction exited before writing 50 statuses' >&2
      return 1
    fi
    sleep 15
  done
}

wait_for_reconstruction
ln -s reconstructions_sparse "$base/reconstructions_ready"
model="$models/Qwen3-VL-8B-Instruct"
# Keep the failed dense run intact and use the successful retry for the repair scripts.
repair_base="$base/repaired_pipeline"
mkdir "$repair_base"
ln -s "$base/observations" "$repair_base/observations"
ln -s "$base/reconstructions_ready" "$repair_base/reconstructions"
run run_abo100_extension prepare-clip --observations "$base/observations" --output "$base/clip_inputs"
run run_abo_image_mass_control predict --dataset "$base/clip_inputs" \
  --checkpoint "$models/open_clip_pytorch_model.bin" --frozen-head "$prior/mass_kg_head.npz" --output "$base/clip"
run run_abo50_mass_hypotheses --observations "$base/observations" --reconstructions "$repair_base/reconstructions" \
  --model "$model" --output "$repair_base/mass_hypotheses"
run replay_abo50_mass_hypotheses --image-only-run "$repair_base/mass_hypotheses" \
  --repair-winding --reconstructions "$repair_base/reconstructions" --output "$repair_base/winding_only"
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
run run_abo100_extension select --cohort "$base/cohort100.json" --previous "$prior/selected_ours" \
  --clip "$base/clip" --construction "$base/panels" --output "$base/selected" "${select_args[@]}"
run run_abo100_extension score --cohort "$base/cohort100.json" --run "$base/selected" \
  --weights "$weights" --nerf "$nerf" --output "$base/score.json"

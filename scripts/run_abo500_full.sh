#!/usr/bin/env bash
set -euo pipefail
root=${1:?experiment root}
models=/root/autodl-tmp/models
prior=/root/autodl-fs/PartPhy3D/abo100_20261120/inputs
py=/root/autodl-tmp/partphy3d-env/bin/python
export PATH="/root/autodl-tmp/partphy3d-env/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
repo=(/root/autodl-tmp/external/InstantMesh-*)
exec 9>"$root/worker.lock"
flock -n 9
for split in val train; do
  n=100
  [[ "$split" == train ]] && n=300
  for ((start=0; start<n; start+=25)); do
    printf -v suffix '%03d' "$start"
    batch="$root/${split}_$suffix"
    until [[ -f "$batch/observations/READY" ]]; do sleep 15; done
    if [[ -f "$batch/role/prediction_seal.json" ]]; then continue; fi
    echo "$(date -Is) $split $start"
    if [[ ! -f "$batch/reconstructions_sparse/progress.json" ]]; then
      "$py" scripts/run_abo50_reconstruction.py --sparse-surface-weights --expected-count 25 \
        --observations "$batch/observations" --repository "${repo[0]}" \
        --checkpoint "$models/InstantMesh-observed/instant_mesh_large.ckpt" \
        --encoder-config configs/pretrained/dino-vitb16 --output "$batch/reconstructions_sparse"
    fi
    "$py" - "$batch" "$prior" <<'PY'
import sys,json
from pathlib import Path
b,p=map(Path,sys.argv[1:])
read=lambda f:json.loads(f.read_bytes())
obs=read(b/'observations/cohort.json')['object_ids']
status=read(b/'reconstructions_sparse/progress.json')['rows']
assert [(r['scene_id'],r['exit_code']) for r in status]==[(s,0) for s in obs]
old=[r['scene_id'] for r in read(p/'selected_ours/predictions.json')]
(b/'cohort75.json').write_text(json.dumps(dict(object_ids=old+obs,prior_object_ids=old,additional_object_ids=obs)))
PY
    if [[ ! -f "$batch/selected/prediction_seal.json" ]]; then
      bash scripts/run_abo100_extension.sh "$batch" "$py" "$models" "$prior" unused unused standalone-batch
    fi
    "$py" scripts/score_abo500_full.py batch "$batch"
  done
done
"$py" scripts/score_abo500_full.py all "$root"

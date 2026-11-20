#!/usr/bin/env bash
# Run the authors' 30-view cached property stage on the other 400 objects.
set -euo pipefail
root=${1:?ABO experiment root}
py=/root/autodl-tmp/partphy3d-env/bin/python
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}" OMP_NUM_THREADS=4
mkdir -p "$root/nerf_run"
exec 9>"$root/nerf_run/worker.lock"
flock -n 9
for ((start=0; start<400; start+=10)); do
  printf -v suffix '%03d' "$start"
  batch="$root/nerf_run/$suffix"
  until [[ -f "$root/nerf_cache/ready/$suffix" ]]; do sleep 20; done
  if [[ -f "$batch/report.json" ]]; then continue; fi
  "$py" -c 'import runpy,sys,torch; torch.cuda.set_per_process_memory_fraction(0.27); sys.path.insert(0,"scripts"); runpy.run_path("scripts/run_nerf_cached_baseline.py",run_name="__main__")' \
    --repository /root/autodl-tmp/external/NeRF2Physics --root "$root/nerf_cache" \
    --clip-checkpoint /root/autodl-tmp/models/open_clip_pytorch_model.bin \
    --weights "$root/weights.json" --cohort "$root/nerf_cache/cohort.json" \
    --start "$start" --limit 10 --resume --output "$batch" > "$root/nerf_run/$suffix.log" 2>&1
  echo "Completed $suffix"
done
"$py" - "$root" <<'PY'
import json,hashlib,sys
from pathlib import Path
root=Path(sys.argv[1])
sets=[root.parent/'abo100_20261120/inputs/nerf']+[root/'nerf_run'/f'{i:03d}' for i in range(0,400,10)]
rows={}
for stage in sets:
    path=stage/'official_predictions.json'
    data=path.read_bytes()
    assert hashlib.sha256(data).hexdigest()==json.loads((stage/'prediction_seal.json').read_bytes())[path.name]
    batch=json.loads(data)
    assert rows.keys().isdisjoint(batch)
    rows.update(batch)
ids=sum(json.loads((root/'splits.json').read_bytes()).values(),[])
assert len(set(ids))==len(rows)==500 and set(ids)==set(rows)
path=root/'nerf_run/predictions.json'
path.write_text(json.dumps(rows,indent=2)+'\n')
(root/'nerf_run/prediction_seal.json').write_text(json.dumps({'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
PY

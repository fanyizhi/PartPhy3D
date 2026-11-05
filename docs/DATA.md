# Data preparation and provenance

The default editing demo is generated in memory from specified rectangular parts.
It needs no dataset download, API key, pretrained checkpoint or GPU.

## ABO-500 diagnostic track

### Prepared five-object subset

The current pilot defaults to `data/abo500_five`, containing the fixed five IDs
in `configs/five_object_pilot.json`. To construct it from an existing local release:

```bash
python scripts/prepare_five_objects.py --source-root /data/abo_500 \
  --mesh-source data/abo500_ten --archive outputs/datasets/ABO500-five.zip
python scripts/prepare_five_objects.py --verify-only
```

```text
data/abo500_five/
  scenes/<scene_id>/images/           original observations, 30 per real object
  scenes/<scene_id>/transforms.json   original camera parameters
  scenes/<scene_id>/info.json         cached upstream annotations/proposals
  meshes_reference/<scene_id>/model.glb
  filtered_product_weights.json      five evaluation labels indexed by scene ID
  splits.json                        original split membership, restricted to five
  selected_objects.json              fixed development selection
  subset_manifest.json               source hashes, label-key mapping, file hashes
  selected_objects.jpg               five-object visual index
```

The builder explicitly maps ASIN-indexed weights to locale-suffixed scene IDs.
Missing or conflicting labels, missing views and invalid cameras fail without
substituting another object. Reference GLBs remain separate from generated Mesh.
It does not include the previous `features/` and `ns/` caches, run inference,
verify metric scale or establish camera-to-reference-Mesh registration.
The data ZIP extracts to `abo500_five/`; extract it into the project's `data/`
directory on the server. Raw data remain local and are excluded from source Git.

These five objects are a development cohort, not SFT training data or an untouched
test benchmark. The standalone synthetic learning diagnostics remain separate.

### Original full release

Obtain the ABO-500 release using the instructions in the upstream
[NeRF2Physics repository](https://github.com/ajzhai/NeRF2Physics).
The helper `bash scripts/download_abo500.sh data` downloads and extracts the
upstream archive; its availability and your data-use rights must be checked
independently. An archive already obtained locally can be extracted instead.

Expected paths relative to the project root:

```text
data/
  abo_500/
    splits.json
    filtered_product_weights.json
    scenes/<scene_id>/
      info.json
      transforms.json
      <image files referenced by transforms.json>
  abo500_ten/
    <scene_id>/model.glb
```

Some ABO-500 PLY files are point clouds with no triangular faces. Native ABO
GLB assets are obtained separately from the
[ABO release](https://amazon-berkeley-objects.s3.amazonaws.com/index.html).
Do not rename a point cloud to make it a surface mesh. Check face count,
coordinates, scene identity, orientation, visibility and metric scale.

The historical development selection was the first ten IDs in the released `test`
list. As of 2026-11-05, the active pilot is the first five IDs of that frozen
selection, stored in `configs/five_object_pilot.json`. The current entry point is
`scripts/run_five_objects.py`; the old ten-object selection and results remain
historical evidence. The helper below downloads their native assets and records per-asset
failures, URLs and hashes; it does not run a learned reconstruction:

```bash
python scripts/prepare_ten_objects.py \
  --root data/abo_500 \
  --assets data/abo500_ten \
  --output outputs/abo500_ten
```

## Generated-mesh main track

Use images with documented scale evidence and the actual frontend's outputs.
Save frontend version, weights identifier, seed, input views and elapsed time.
Keep reference and generated meshes in separate experiment directories. Native
ABO bounding boxes or volumes must not silently supply scale to the main track.
The complete InstantMesh/Qwen/learned-part path has not been executed here.

## Material-color preview

`scripts/preview_material_edit.py` currently expects the separately prepared
ottoman asset and matching view at:

```text
data/abo500_ten/B082VML3QJ_ATVPDKIKX0DER/model.glb
data/material_color_demo/source.png
data/material_color_demo/source.json
data/material_color_demo/camera.json
```

Use `python scripts/preview_material_edit.py --help` for overrides. The camera
record must contain `w,h,fl_x,fl_y,cx,cy,camera_to_world` and match the selected
image exactly. This object-specific color preview is not a general segmentation
model and does not commit a physical edit or estimate absolute object mass.

## Labels, models and redistribution

Keep mass labels outside inference prompts and construction selection. Evaluation
reads them only after prediction sealing. Density, modulus, friction and edited
mass need their own independently sourced reference labels; ABO total weight
alone does not establish them. Material presets are hypotheses, not measurements.

No dataset, product photo, upstream model or checkpoint is bundled in the source release. The MIT
license applies to this project's code, not third-party data or weights. Keep
API credentials in the local environment and outside Git. See
[THIRD_PARTY.md](../THIRD_PARTY.md) for upstream resources.

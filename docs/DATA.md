# Data preparation and provenance

The default editing demo is generated in memory from specified rectangular parts.
It needs no dataset download, API key, pretrained checkpoint or GPU.

## ABO-500 diagnostic track

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

The fixed development selection is the first ten IDs in the released `test`
list. The helper below downloads only their native assets and records per-asset
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

No dataset, product photo, upstream model or checkpoint is bundled. The MIT
license applies to this project's code, not third-party data or weights. Keep
API credentials in the local environment and outside Git. See
[THIRD_PARTY.md](../THIRD_PARTY.md) for upstream resources.

# ABO-500 five-object development subset

Git tracks metadata, camera files, selection manifests and licenses only.
The 150 original views and native reference GLBs are local dataset files.
They are not included in a fresh Git clone.
scenes/: cameras and cached upstream info.json; images/ must be prepared locally.
meshes_reference/: local native reference GLBs, not InstantMesh outputs.
filtered_product_weights.json: evaluation labels, excluded from prompts.
Scale and camera-to-GLB registration are not verified by copying files.
No features/, ns/ caches or inferred part/physical labels are included.
This is a development cohort, not an untouched test benchmark.
Upstream data licenses apply; the project MIT license covers code only.

See ../../docs/DATA.md for preparation commands. After restoring the data, run
python scripts/prepare_five_objects.py --verify-only.

# PartPhy3D

[![CPU tests](https://github.com/fanyizhi/PartPhy3D/actions/workflows/tests.yml/badge.svg)](https://github.com/fanyizhi/PartPhy3D/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Construction-aware physical inference and part editing. **Research prototype,
version 0.6.0; no peer-reviewed performance or acceptance claim.**

[中文](README_zh.md) · [Complete methods and reproduction](docs/METHODS_AND_REPRODUCTION_zh.md) ·
[Status](docs/IMPLEMENTATION_STATUS.md) · [Data](docs/DATA.md) · [Evidence](docs/results/README.md)

As of 2026-11-05, the current real-object pilot uses **five development objects**,
the first five IDs of the previously frozen ten-object cohort. Use
`scripts/run_five_objects.py`; historical ten-object results are retained.
The pilot now defaults to `data/abo500_five/`. Build this portable subset with
`scripts/prepare_five_objects.py`: original views, cameras, separate reference GLBs,
five evaluation labels and SHA-256 checks. See [data preparation](docs/DATA.md).

**New Mesh-native path:** disjoint closed material regions, signed boundary
integration, cached mass/COM/full inertia and fixed-construction material edits.
No voxelization is called by this path. The voxel prototype remains available.
See [the detailed implementation guide](docs/mesh_native_v0.6_zh.md).

## Reproduce on CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,figures]"
python -m pytest -q
python scripts/reproduce_mesh_native.py --output outputs/mesh_first
```

Linux CI checks Python 3.10; development also uses Python 3.13. On Windows activate
with `.venv\Scripts\Activate.ps1`. No GPU, torch, dataset download or SSH is used
by the default suite. Output directories are never overwritten.

The suite generates 384 analytic objects, splits **by object family** into
246 training, 66 calibration and 72 test objects, trains three small NumPy models
with three initializations, and verifies Mesh-native material editing against
independent box/cavity references. **These are synthetic diagnostics, not ABO/VLM results.**

```text
outputs/mesh_first/
  suite.json                         # status, scope, remaining research work
  editing/mesh_edit_views.png         # synchronized material colors
  editing/mesh_property_sections.png  # Mesh sections: density, fill, modulus
  editing/physics.csv                 # mass, COM, complete inertia
  editing/original/part_tokens.json    # executable schema, not learned embeddings
  learning/metrics.csv                # original and edited mass metrics
  learning/learning_results.png        # actual synthetic training and evaluation
```

The known-design reference mass is 9.7728 → 25.1136 kg. It verifies software;
it does not establish real-object inference accuracy. The learned encoder still
uses 11 synthetic proxy features. The legacy voxel/observability/adaptive suite
is retained as `python scripts/reproduce_all.py --output outputs/first_run`:

```text
outputs/first_run/
  suite.json                         # status and elapsed time
  ARTIFACTS_SHA256.json               # exported artifact hashes
  learning/
    dataset/                         # separate inputs and labels
    seed_17/                         # checkpoints, sealed predictions, states
    metrics.csv                      # original, edited, MAP-asset mass errors
    state_metrics.csv                # density, occupancy, construction, soft/MAP gap
    test_predictions.csv             # raw predictions and calibrated intervals
    training.csv
    learning_results.png             # also SVG
    physical_fields.png              # density and occupancy sections, also SVG
  editing/                           # material, construction, donor edit replay
  adaptive/                          # meshes, cost/error curves and CSV
  observability.json
```

## What works

- Mesh-native scalar-material regions, closed-shell Mesh CSG, signed geometric
  moments, overlap checks, unit validation and replayable material edits.
- Explicit Object/Part/Region/Cell state with SI units, identity, versions, void,
  evidence and mounting interfaces.
- Mass, COM and full inertia from one density field, including finite-cell inertia;
  atomic material, closed-box and compatible donor edits, replay and cache checks.
- Differentiable finite-construction diagnostic: shared state, original mass loss,
  signed edit-response loss, construction supervision, analytic gradients,
  mass-only and direct-regression controls. The encoder uses **11 proxy features**,
  not pixels or VLM embeddings.
- Family-held-out mass calibration and separate soft-mixture/MAP asset evaluation.
- Conforming task-weighted coarse-surface refinement using a supplied field. The
  bundled field is analytic; this is **not an adaptive InstantMesh decoder**.
- ABO reader, geometry-clustering baseline, material priors, optional local
  Qwen-compatible HTTP inference, mass metrics and figure tools.
- Official InstantMesh external command adapter: read-only checks by default,
  explicit metric scale, hashes, logs and retained failures.

The complete image-to-learned-part-state method, fine semantic segmentation,
real paired edit supervision and robot/MPM validation remain research work.
No CPU diagnostic is silently substituted for those experiments.

## Real ABO and models

Data and pretrained weights have separate licenses. Follow the
[full guide](docs/METHODS_AND_REPRODUCTION_zh.md).
ABO-500 PLY files may be point clouds. Reference GLBs and generated meshes must be
recorded as different input conditions. Metric scale and camera registration are
required. Example two-object **rule baseline**, with explicit approximate repair:

```bash
python -m tactile3d.cli abo500-run \
  --root /data/abo_500 --mesh-root /data/abo_meshes \
  --mesh-unit-to-meter 1 --split test --limit 2 \
  --config configs/abo500_envelope_smoke.yaml --output outputs/abo_two
python -m tactile3d.cli abo500-evaluate \
  --predictions outputs/abo_two/predictions.json --output outputs/abo_two/metrics.json
```

Use `1` only for independently verified meter-scale meshes. Envelope filling does
not reveal true interiors. Strict repair in `configs/abo500_mesh_smoke.yaml` fails
rather than silently using a convex hull. The executed two-object rule baseline
has large errors; see the [report](docs/results/reproduction_v0.5.0_zh.md).

## Other commands

Version 0.5.1 binds material-region selection, schematic colors, density and
declared elastic properties to the same state. Run the known-design chair example:

```bash
python scripts/edit_material.py --output outputs/material_first
```

This exports three before/after views, physical changes, editable assets and
independent analytic checks. See [material editing](docs/material_edit_v1_zh.md).
The full CPU suite includes this experiment under `bound_material/`.

```bash
python scripts/reproduce.py --output outputs/learn_only
python scripts/construct_adaptive.py --output outputs/construct_only
python scripts/edit_parts.py --plan examples/wood_to_hollow_steel.json
python scripts/audit_edit_observability.py
python scripts/reconstruct_instantmesh.py --help
python -m tactile3d.cli --help
```

## Package and install

```bash
python scripts/package_project.py --output outputs/release --results outputs/mesh_first
```

Produces an audited source ZIP, wheel, sdist, optional CPU-results ZIP and SHA-256
checksums. Excludes datasets, weights, credentials and local SSH configuration.
Dependencies are **not** an offline wheelhouse. After installing the wheel, run
`partphy3d-mesh-reproduce --output PATH` from any directory.

`requirements.txt` is the single full project dependency list. The CPU quickstart
uses optional dependencies to avoid downloading training packages. Upstream model
installations have separate compatibility requirements not verified by CPU tests.

## Evaluation and integrity

ADE: kg; ALDE: natural log; APE: fraction; MnRE: higher is better. Mass intervals
are calibrated on held-out families, not confidence in hidden construction.
Unlabeled modulus, friction and hardness maps are priors, not measured accuracy.
PUGS, NeRF2Physics, VoMP/AdaVoMP, Gen2Physics and PhysVGGT remain pending until
actual predictions are reproduced under matched inputs and supervision.
Published aggregate scores are not mixed into reproduced tables.

See [evaluation](docs/evaluation_protocol_zh.md), [documentation](docs/README.md),
[MIT license](LICENSE), [third-party terms](THIRD_PARTY.md), [citation](CITATION.cff)
and [contribution guide](CONTRIBUTING.md).

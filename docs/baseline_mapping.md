# Baseline-to-Mesh Mapping

This project uses the official MIT-licensed PUGS and NeRF2Physics releases as
methodological references. It does not vendor their NeRF, Gaussian rasterizer,
CUDA submodules, model weights, datasets, or generated intermediates.

| Baseline operation | Mesh-first counterpart |
| --- | --- |
| NeRF/3DGS reconstruction | Supplied triangular mesh with explicit metric scale |
| Extracted points/Gaussians | Mesh faces and filled occupancy voxels |
| CLIP feature fusion | Occlusion-aware multi-view face-part crops |
| Candidate material ranges | Physical priors plus structured VLM proposal |
| Surface thickness heuristic | Solid/shell/layered/core hypothesis integration |
| Gaussian pure-volume correction | Exact enclosed mesh volume and sampled occupancy |
| Low/high output midpoint | Posterior mean and q05/q50/q95 |

## Reused Concepts

- Multi-view language-aligned evidence for material assignment.
- Candidate material properties instead of a closed-set scalar regressor.
- Spatial integration of density to obtain object mass.
- ABO-500 ADE, ALDE, APE, and MnRE evaluation.

## Replaced Components

- No per-scene NeRF or 3DGS optimization is required.
- Camera projections target explicit mesh faces.
- Mesh volume is never inferred from unscaled normalized coordinates.
- Hidden construction is a distribution over solid, hollow, layered,
  foam-filled, frame-supported, and partial-infill hypotheses.
- Cached structured JSON makes VLM inference auditable and repeatable.

## Identifiability

RGB images and an exterior mesh do not uniquely determine hidden construction.
ABO-500 object mass can calibrate aggregate predictions but cannot supervise a
unique per-part bill of materials. Hidden material, density, and Young's modulus
must be reported as inferred hypotheses unless CT, CAD, product BOM, acoustic,
deformation, or tactile evidence is available.


## Locked mass reproduction (2026-11-11)

The executable entry is `scripts/run_nerf_cached_baseline.py`. Source hashes and
CLIP weights are fixed by `configs/nerf2physics_lock.json`. The complete published
100-object test cohort is fixed in `configs/abo500_mass_test.json`; the first ten
objects were previously exposed during development. Report both the original
100-object protocol and the remaining 90-object subset, with no test-time tuning.

For NeRF2Physics, CLIP produces material probabilities
$p_{ij}=\mathrm{softmax}_j(\langle f_i,t_j\rangle/0.1)$.
For native lower/upper endpoint $b$, the thickness branch sums
$u^{(b)}=\sum_{ij}p_{ij}\rho_j^{(b)}a_i t_j^{(b)}$,
where $a_i=(0.005/s)^2$ and $s$ is the released camera normalization scale.
The official code uses
$\widetilde V=\sum_i\max_b\sum_jp_{ij}a_i t_j^{(b)}$,
then returns $m^{(b)}=0.6\,u^{(b)}\min(1,V_{carve}/\widetilde V)$.
Density and thickness endpoints are paired by the upstream implementation;
these bounds are material ranges, not probability quantiles.
The original depth maps and all 30 alpha-masked views participate in carving.

PUGS uses $v_i=(4\pi/3)\prod_k\sigma_{ik}\,\alpha_i\,(0.6826)^3$
for Gaussians retained at opacity $\alpha_i>0.95$.
Its Gaussian branch computes
$m^{(b)}=V_{pure}\,\sum_{ij}v_i p_{ij}\rho_j^{(b)}/\sum_i v_i$.
Thus the common volume multiplier cancels, while the VLM-proposed pure material
volume remains essential. This differs from integrating density over the
exterior bounding volume. Reusing NeRF points or replacing SAM ViT-H with SAM2
would be a modified baseline, not the released PUGS pipeline.

Both evaluation scripts use $\hat m=(m^{(low)}+m^{(high)})/2$,
then macro-average ADE, natural-log ALDE, APE and MnRE as defined in
[evaluation protocol](evaluation_protocol_zh.md). The runner calls the original
NeRF2Physics evaluation functions and checks agreement to $10^{-12}$ before
saving a final report. Predictions and input hashes are sealed before opening
mass labels. Runtime and CUDA memory cover property inference only.

PartPhy3D should compare whole-object $m=\sum_r\rho_r V_r$ on a metric,
non-overlapping material partition. Closed-leg-only integration belongs in the
local editing study and cannot enter a whole-object mass ranking. The intended
contribution is how evidence constrains material occupancy and how simplification
preserves its integrals; the density integral itself is established physics.

Sources: [NeRF2Physics integration](https://github.com/ajzhai/NeRF2Physics/blob/7311210f62b120f6896b3112b231c26003ce70e3/predict_property.py),
[NeRF2Physics evaluation](https://github.com/ajzhai/NeRF2Physics/blob/7311210f62b120f6896b3112b231c26003ce70e3/evaluation.py),
[PUGS integration](https://github.com/EverNorif/PUGS/blob/ef7fd8c52543ab0fe6eb2a7a99e5267523133a09/predict_property.py).


### Run the cached property stage

```bash
python scripts/run_nerf_cached_baseline.py \
  --repository /path/to/NeRF2Physics \
  --root /path/to/abo_500 \
  --clip-checkpoint /path/to/open_clip_pytorch_model.bin \
  --weights /path/to/abo_500/filtered_product_weights.json \
  --cohort configs/abo500_mass_test.json \
  --output outputs/nerf2physics_test100
```

`--start` and `--limit` select contiguous batches without changing their order;
`--resume` verifies saved input hashes before reusing each prediction. Omitting
`--cohort` uses the frozen five-object development pilot. `--mesh-manifest` is
optional and adds a separate generated-mesh field query. Upstream dependencies
include OpenCLIP, Open3D, OpenAI (import only), PyTorch and Matplotlib; no API calls
are made in this cached mode. PUGS source and SAM hashes are separately recorded
in `configs/pugs_lock.json`; this lock alone does not indicate a completed run.


### PUGS with the existing local VLM

`scripts/run_pugs_baseline.py` provides `prepare`, `reconstruct`, `propose` and
`integrate` stages. Use the PUGS environment for reconstruction/integration and
the existing local-Qwen environment for proposal generation. All stages require
`--repository`, `--scene` (the original 30-view scene directory), and `--output`.

- `prepare` exports the dimension-only catalog to the upstream
  `filtered_metadata.json` layout. Meters become centimeters; preferred-domain
  records are used when available, otherwise a unique available record is used.
  Existing inconsistent metadata is rejected. No mass labels are exported.
- `reconstruct --sam-checkpoint PATH` runs the official 30,000-step Gaussian
  reconstruction, SAM ViT-H masks, mask scales and 10,000-step contrastive feature
  training. The five CUDA extensions and PyTorch3D 0.7.7 must be built first.
- `propose --model PATH` uses the exact enhanced density/pure-volume system
  prompt from the locked upstream file. This local variant uses one fixed view
  (index 8), NF4 Qwen3-VL-8B, greedy decoding, seed 100, at most 300 output tokens
  and 1024-squared pixels. The minimal user text is `Analyze the image.`.
  Raw responses and model settings are saved before parsing; `info_qwen.json`
  preserves the original cached `info.json`. Candidate count follows the upstream
  parser, which accepts variable counts despite the five-entry schema example.
- `integrate --clip-checkpoint PATH` calls the official CLIP feature-fusion and
  Gaussian-volume integration functions. Native ranges are sealed before labels
  are opened. It is recorded as **PUGS-Qwen3-VL-8B-NF4**, never official GPT-4o.

The initial PUGS environment uses PyTorch 2.1.2+cu118, CUDA toolkit 11.8,
PyTorch3D 0.7.7 and NumPy 1.26.4. Python is 3.10 rather than the upstream 3.9;
OpenCLIP is 2.24.0. These environment differences are recorded with the results.
Official GPT-4o remains unexecuted without API configuration. The local variant
uses existing weights and makes no paid model calls. One development object is
an integration check; it cannot supply a full-cohort comparison score.


### Executed mass results

The 2026-11-11 NeRF2Physics released-cache property run completed all 100 published
test objects. All ten batches matched the author's four evaluation functions to
numerical tolerance. Full source data and 2,000-resample, seed-17 percentile
bootstrap intervals are in [the result record](results/mass_reproduction_20261111.json).

| Cohort | N | ADE kg ↓ | ALDE ↓ | APE % ↓ | MnRE ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published test split | 100 | 9.8709 | 0.7787 | 114.7926 | 0.5511 |
| Excluding the ten previously exposed development objects | 90 | 10.4775 | 0.7850 | 122.0956 | 0.5492 |

For the full 100-object cohort, the ADE 95% interval is [6.6999, 13.5138] kg.
The released mass ranges are not q05/q95 intervals. Runtime records cover only
the property stage and include periods with concurrent development jobs, so they
are not an end-to-end speed comparison. These are rerun results, not copied
published numbers. PUGS-Qwen currently has one development object; its complete
100-object comparison remains pending.


### Material/geometry integration audit (2026-11-12)

A paired audit reuses the five frozen development objects, the same 144-resolution
cell envelopes, geometry/color regions, cached material priors, construction
fractions and random seed. The historical prototype averaged candidate thickness
before sampling density. This loses the material/thickness dependence:
`E[rho * V_shell(t)]` generally differs from `E[rho] * V_shell(E[t])`.
The corrected integrator conditions shell volume and all core/outer fields on
the same material draw. Its cell-field mean equals the sampled mass integral.
Within-material thickness uncertainty is still held fixed at its nominal value.

A separate geometry ablation replaces nearest triangle-centroid ownership with
batched closest-triangle-surface queries. This prevents large-face centroid bias;
it is a geometric assignment rule, not a recovery of hidden internal anatomy.
The historical `predict` mode retains its original integration default. The new
`audit-shell` stage explicitly evaluates all three alternatives.

| Same five development objects | ADE kg ↓ | ALDE ↓ | APE % ↓ | MnRE ↑ |
| --- | ---: | ---: | ---: | ---: |
| Historical rule prototype | 9.2105 | 0.7281 | 96.8668 | 0.4957 |
| Material-conditioned shell | 7.2582 | 0.6178 | 72.0859 | 0.5554 |
| Material-conditioned shell + nearest surface | 7.2130 | 0.6203 | 72.3729 | 0.5534 |
| NeRF2Physics cached stage, same five | 2.9598 | 0.5122 | 47.7386 | 0.6226 |

The nearest-surface variant reduces ADE by 21.69% relative to the historical
prototype; its ALDE, APE and MnRE are slightly worse than conditioning alone.
The two corrected variants trade off these metrics. The multi-round VLM full method remains
a separate pending whole-object comparison. Old color-driven region/material
rules and generic hidden-fill assumptions remain major targets for replacement.
[Predictions, frozen configuration and seal](results/mass_integration_audit_20261112.json)
record the complete five-object audit; no mass fitting or object substitution was
performed. This development audit is not added to the 100-object published-test table.

```bash
python -m pip install -e ".[study]"
python -m tactile3d.experiments.ten_objects \
  --root /path/to/abo_500 \
  --assets /path/to/reference_meshes \
  --stage audit-shell \
  --cohort configs/five_object_pilot.json \
  --output outputs/material_geometry_audit
```

The reference-mesh folder must contain `<scene_id>/model.glb`. The stage seals
all predictions and fields before opening mass labels, retains unsuccessful
objects in the requested cohort, and reports aggregate scores only when complete.

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

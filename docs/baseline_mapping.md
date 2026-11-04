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

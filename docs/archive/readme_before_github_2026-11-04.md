# PartPhy3D: Mesh Physical Grounding and Editing

**Generation-stage scope and material-color preview (2026-11-04):**
[`docs/generation_and_material_preview_2026-11-04_zh.md`](docs/generation_and_material_preview_2026-11-04_zh.md).
Run `python scripts/preview_material_edit.py` for the existing ABO ottoman's
reference-mesh color preview. Generation-stage adaptive construction remains a
research target; current QEM postprocessing does not accelerate the frontend.

**Detailed method review (2026-11-04):**
[`docs/method_step_review_2026-11-04_zh.md`](docs/method_step_review_2026-11-04_zh.md)
explains each step, proposed improvements, required evidence, limitations, and
the boundary between executed software and pending research mechanisms.

**Editing-command executor (2026-11-04):**
[`docs/edit_commands_2026-11-04_zh.md`](docs/edit_commands_2026-11-04_zh.md).
Run `python scripts/edit_parts.py` for a complete CPU synthetic validation with
explicit Chinese instructions, strict JSON programs, transaction rollback,
portable donor snapshots, and physical/geometry/density figures. This validates
the editing software; it does not run a learned model or establish real-object
physical accuracy.

**Executed fixed-cohort comparison entry (2026-11-03):**
[`docs/comparison_implementation_2026-11-03_zh.md`](docs/comparison_implementation_2026-11-03_zh.md).
Run `python scripts/compare_experiments.py` to evaluate the sealed ten-object
rule baselines and export tables/figures. Published baselines and the learned
full pipeline remain explicitly pending; missing methods receive no scores.

**Executed local part-state and editing prototype (2026-11-03):**
[`docs/part_state_implementation_2026-11-03_zh.md`](docs/part_state_implementation_2026-11-03_zh.md).
Run `python scripts/demo_part_edit.py` for the synthetic numerical demo.
Material/structure edits and compatible voxel donors share one physical state;
21 related checks and a ten-object cached-field audit passed. These are software
checks, not learned-model or real-object accuracy results.

**Current step-by-step pipeline (2026-11-03):**
[`docs/current_pipeline_2026-11-03_zh.md`](docs/current_pipeline_2026-11-03_zh.md).
Use InstantMesh as a fixed frontend and focus on physical inference, constrained
part editing, and task-aware mesh simplification. Custom image reconstruction is
an alternative extension. The executed ten-object batch still uses original ABO
reference assets; the real-model pipeline remains pending.

**Current research scope and next-stage acceptance (2026-11-02):**
[`docs/research_agreement_2026-11-02_zh.md`](docs/research_agreement_2026-11-02_zh.md).
Validate the real-model pipeline on the fixed ten development objects before
expanding to the full dataset; distinguish executed rule baselines from pending models.

The executed **10-object development batch** is documented in
[`docs/ten_object_experiment_2026-10-30_zh.md`](docs/ten_object_experiment_2026-10-30_zh.md).
All ten objects were processed with shared untrained rules; the report retains
large errors, geometry sensitivity, and missing local-property references.
This batch has not executed InstantMesh, Qwen, learned part segmentation or paper baselines.

The executed ABO-500 one-object experiment (2026-10-30), its measured mass
metrics, figures, reproducible commands, and limitations are in
[`docs/single_object_experiment_2026-10-30_zh.md`](docs/single_object_experiment_2026-10-30_zh.md).
This uses the first released test object, a reference mesh and dataset silhouettes
with untrained construction priors. It does not claim trained-model or published
baseline performance. The current paper method is specified in
[`docs/ieee_trans_method_contributions_zh.md`](docs/ieee_trans_method_contributions_zh.md).

> **Fixed project snapshot (2026-09-15):** the complete Chinese summary of the
> research idea, decisions, implementation, metrics, and next steps is preserved
> in [`docs/session_snapshot_2026-09-15_zh.md`](docs/session_snapshot_2026-09-15_zh.md).
> The executable Chinese reproduction checklist is in
> [`docs/reproduction_zh.md`](docs/reproduction_zh.md).
> The physics-aware editing paper story and implementation contract are in
> [`docs/physics_editing_story_zh.md`](docs/physics_editing_story_zh.md).
> Server capability and smoke-test instructions are in
> [`docs/server_testing_zh.md`](docs/server_testing_zh.md).

Research code for probabilistic, part-level physical property inference and
physics-aware editing of metric textured meshes. It supports both the ABO-500
benchmark protocol and mesh-only inference with internally rendered views.

The scientific motivation, hypotheses, data hierarchy, psychophysical protocol,
ablation plan, and honest implementation-status table are documented in
[`docs/research_design_zh.md`](docs/research_design_zh.md).

This repository implements a mesh counterpart to NeRF2Physics and PUGS. It
projects a repaired mesh into every ABO-500 camera, aggregates part-level visual
material evidence, proposes visible and latent interior materials, predicts
distributions over density and Young's modulus, and integrates density over a
closed metric volume to estimate mass.

> Scope: ABO-500 provides object-level ground-truth mass, not measured per-part
> density or Young's modulus. Those properties remain probabilistic material
> estimates and must not be reported as measured ground truth.

## Method

For RGB views and calibrated cameras
\(\mathcal{I}=\{(I_v,K_v,T_v)\}_{v=1}^{V}\), a metric mesh \(M\), and part
labels \(S\), the model estimates

\[
p(\rho_k,E_k,z_k,t_k,\phi_k\mid \mathcal{I},M,S),
\qquad
m=\sum_k m_k.
\]

Here (z_k) is an interior construction hypothesis, (t_k) is shell
thickness, and \(\phi_k\) is occupied-core fraction. A sampled part mass is

\[
m_k=\begin{cases}
\rho_kV_k & \text{solid},\\
\rho_kA_kt_k & \text{hollow shell},\\
\rho_sV_s+\sum_j\rho_j\phi_jV_j & \text{layered or filled}.
\end{cases}
\]

The implemented pipeline is:

1. Load the official ABO-500 split, camera poses, images, and mass target.
2. Repair the mesh and reject it if a closed volume cannot be established.
3. Segment faces using geometry and adjacency-aware smoothing.
4. Fill the mesh with voxels and assign interior voxels to surface parts. The
   resulting fractions divide the exact enclosed mesh volume.
5. Project each part into all cameras with a coarse z-buffer and retain the most
   informative views.
6. Score visible surface materials with OpenCLIP.
7. Optionally query a structured multi-view VLM for surface-material and hidden
   solid, hollow, layered, foam, frame, infill, and void hypotheses.
8. Fuse OpenCLIP and VLM surface evidence, then sample construction, wall
   thickness, fill ratio, material density, modulus, and metric scale.
9. Integrate each sample and report a direct mass estimate plus calibrated
   5/50/95 posterior percentiles.

ABO-500 train masses can weakly calibrate material logits, global density scale,
temperature, and residual uncertainty. Validation and test labels are never used
by the fitting command.

## Repository Layout

```text
configs/abo500_mesh.yaml          GPU/OpenCLIP experiment configuration
configs/abo500_mesh_vlm.yaml      local/hosted multimodal VLM configuration
configs/abo500_mesh_smoke.yaml    CPU test configuration
scripts/download_abo500.sh        official ABO-500 downloader
scripts/run_abo500_mesh.sh        train, validation, and test workflow
scripts/run_abo500_mesh_vlm.sh    mesh + multi-view VLM workflow
src/tactile3d/abo500/             dataset reader and full pipeline
src/tactile3d/vision/             mesh projection, material encoder, VLM client
src/tactile3d/physics/interior.py probabilistic interior-volume integration
src/tactile3d/models/             weakly supervised mass calibration
src/tactile3d/eval/               ABO-500 metrics
tests/                             unit and end-to-end synthetic tests
```

Additional research modules are:

```text
src/tactile3d/mesh/render_blender.py mesh-only headless rendering bridge
src/tactile3d/mesh/editing.py        retrieval-based part replacement baseline
src/tactile3d/training/dataset.py    expert/professional multimodal SFT builder
src/tactile3d/training/psychophysics.py human pairwise-scale estimation
src/tactile3d/training/train_vlm.py  Qwen3-VL LoRA/QLoRA entry point
```

## Installation

Linux with an NVIDIA GPU is recommended. The single environment includes the
core pipeline, OpenCLIP, QLoRA training, publication figures, and tests.

```bash
conda env create -f environment.yml
conda activate partphy3d
pytest -q
```

## Mesh-Only Inference

Provide a textured mesh and the metric length represented by one mesh unit.
Blender renders deterministic orbit views internally, so no external images or
camera poses are required:

```bash
bash scripts/run_mesh_only.sh object.glb 0.001 outputs/object
```

This requires Blender on `PATH`; an absolute executable path can be supplied as
the fourth argument. A textureless mesh cannot provide reliable visual material
evidence and should be treated as a geometry-only ablation.

Inspect the enclosed volume before physical inference:

```bash
tactile3d mesh-volume \
  --mesh object.glb \
  --mesh-unit-to-meter 0.001 \
  --output outputs/object/volume.json
```

For a uniform material, `--density 7800 --fill-fraction 1.0` also reports the
corresponding mass. Non-watertight inputs return no exact volume unless the
explicit `--convex-hull-proxy` approximation is requested.

## VLM Fine-Tuning Data

Prepare object-grouped train/validation/test splits from expert JSONL records:

```bash
tactile3d prepare-sft \
  --annotations data/annotations/expert_parts.jsonl \
  --image-root data/renders \
  --output data/sft

bash scripts/train_vlm_lora.sh data/sft outputs/qwen3vl_lora
```

For two RTX 4090 24 GB cards, prepare no more than four views per sample and
use the distributed QLoRA launcher:

```bash
tactile3d prepare-sft \
  --annotations data/annotations/expert_parts.jsonl \
  --image-root data/renders \
  --max-images 4 \
  --output data/sft

bash scripts/train_vlm_2x4090.sh data/sft outputs/qwen3vl_8b_qlora
```

The launcher uses batch size 1 per GPU, gradient accumulation 8, SDPA,
gradient checkpointing, and a `384 * 28 * 28` per-image pixel cap. Reduce
`--max-images` to 2 or the pixel cap to `256 * 28 * 28` if a particular driver
or long JSON target still causes CUDA out-of-memory.

See `docs/expert_annotation.example.jsonl` and
`docs/psychophysics.example.jsonl` for the expected source schemas. Replace the
example records with licensed measurements and ethics-approved annotations;
they are format examples, not training evidence.

Fit a human perceptual scale from pairwise judgements without using held-out
rows for fitting:

```bash
tactile3d fit-psychophysics \
  --comparisons data/psychophysics/expected_weight.jsonl \
  --output outputs/expected_weight_scale.json
```

## Physics-Aware Part Replacement

The first editing baseline replaces a face-labelled source part with a donor,
then reports mass and centre-of-mass changes, static support, interface gap,
watertightness, and whether the output remains a disconnected assembly:

```bash
tactile3d part-replace \
  --source outputs/object/result/mesh_metric.ply \
  --face-labels outputs/object/result/face_labels.npy \
  --target-part 2 \
  --donor data/donors/metal_leg.ply \
  --mesh-unit-to-meter 1.0 \
  --source-density 650 \
  --donor-density 7850 \
  --output outputs/object/edit_leg
```

This command is a transparent retrieval baseline, not yet a learned generative
editor or a guaranteed manufacturing-quality boolean union.

Run a complete five-object synthetic smoke experiment with the same directory
and command interfaces as ABO-500:

```bash
bash scripts/run_abo500_smoke.sh
```

## Structured Interior VLM

The default `abo500_mesh.yaml` is the all-solid ablation and does not contact a
VLM. `abo500_mesh_vlm.yaml` targets any OpenAI-compatible multimodal endpoint,
including a local Qwen3-VL model served by vLLM. Start the server separately,
then run:

```bash
export TACTILE3D_VLM_BASE_URL=http://127.0.0.1:8000/v1
export TACTILE3D_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

bash scripts/run_abo500_mesh_vlm.sh \
  data/abo_500 \
  data/abo_500_meshes \
  0.001 \
  outputs/abo500_mesh_vlm
```

For a hosted endpoint, also set `TACTILE3D_VLM_API_KEY`. Credentials are read
only from the environment and are never stored in output artifacts.

The VLM receives one full-object context image, several projected part crops,
and metric part geometry. It must emit strict JSON containing:

- probabilities over visible surface materials;
- 2-4 interior construction hypotheses;
- shell thickness intervals;
- hidden core/frame/infill materials and occupied-volume fractions.

Each scene saves normalized proposals in `interior_proposals.json`. Reuse those
responses without another model call:

```bash
tactile3d abo500-run \
  --root data/abo_500 \
  --mesh-root data/abo_500_meshes \
  --mesh-unit-to-meter 0.001 \
  --config configs/abo500_mesh_vlm.yaml \
  --vlm-backend file \
  --vlm-response-root outputs/abo500_mesh_vlm/train/scenes \
  --split train \
  --output outputs/abo500_cached/train
```

Use `--vlm-backend surface_only` for the visible-material/full-solid ablation,
or `--vlm-backend heuristic` for an offline integration test. VLM interior
output is a prior, not an observation of hidden matter.

## ABO-500 Data

Download the release provided by NeRF2Physics:

```bash
bash scripts/download_abo500.sh data
```

Expected structure:

```text
data/abo_500/
  filtered_product_weights.json
  splits.json
  scenes/
    <scene_id>/
      images/
      transforms.json
```

The official benchmark uses 500 objects with 30 views per object and a fixed
300/100/100 train/validation/test split.

## Mesh Preparation

The original [Amazon Berkeley Objects release](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
does include more than 7,900 textured 3D models in glTF 2.0 format. The project
reader supports `.glb` directly, in addition to PLY, OBJ, STL, and OFF.

The NeRF2Physics ABO-500 bundle explicitly documents images, cameras, weights,
depth maps, and point clouds, but does not document a ready-to-use one-to-one
GLB layout for this mesh pipeline. Match an original ABO model through verified
metadata such as `3dmodel_id`, or use a reconstructed mesh, then stage it as:

```text
data/abo_500_meshes/<scene_id>.glb
```

`<scene_id>.ply` and the other supported extensions are also accepted. Do not
infer the mapping from an ASIN-looking scene name alone: a public NeRF2Physics
issue reports examples where an ABO-500 scene label and the original ABO listing
under that ASIN describe different products. Inspect the rendered mesh against
the scene images before accepting the pair.

The mesh must also share the world coordinate frame used by `transforms.json`.
Its length unit must be verified rather than inferred from the file format.
Either pass one global conversion factor:

```bash
tactile3d abo500-validate \
  --root data/abo_500 \
  --mesh-root data/abo_500_meshes \
  --mesh-unit-to-meter 0.001
```

or add per-scene metadata:

```json
{"mesh_unit_to_meter": 0.001}
```

at `scenes/<scene_id>/mesh_scale.json`.

`mesh_unit_to_meter=0.001` means one mesh unit is one millimetre. Never use an
arbitrary value for a normalized NeRF or generated mesh: volume changes with the
cube of scale, so mass is not identifiable until metric scale is recovered from
known dimensions, a reference object, depth, or camera calibration.

## Full Experiment

Run feature extraction on train, fit the mass calibrator, then evaluate val/test:

```bash
bash scripts/run_abo500_mesh.sh \
  data/abo_500 \
  data/abo_500_meshes \
  0.001 \
  outputs/abo500_mesh
```

Equivalent individual commands:

```bash
tactile3d abo500-run \
  --root data/abo_500 \
  --mesh-root data/abo_500_meshes \
  --mesh-unit-to-meter 0.001 \
  --split train \
  --output outputs/abo500_mesh/train

tactile3d abo500-calibrate \
  --features outputs/abo500_mesh/train/features.jsonl \
  --output outputs/abo500_mesh/mass_calibration.json

tactile3d abo500-run \
  --root data/abo_500 \
  --mesh-root data/abo_500_meshes \
  --mesh-unit-to-meter 0.001 \
  --split test \
  --calibration outputs/abo500_mesh/mass_calibration.json \
  --output outputs/abo500_mesh/test

tactile3d abo500-evaluate \
  --predictions outputs/abo500_mesh/test/predictions.json \
  --output outputs/abo500_mesh/test/metrics.json
```

Use `--limit 2` with `abo500-run` for a quick integration test. The first
OpenCLIP run downloads its public checkpoint.

## Outputs

Each `outputs/<split>/scenes/<scene_id>/` directory contains:

- `mesh_metric.ply`: repaired mesh in metres.
- `mesh_parts.ply`: colour-coded part segmentation.
- `face_labels.npy`: integer part label for every face.
- `interior_proposals.json`: normalized VLM construction hypotheses.
- `prediction.json`: part and object posterior distributions.

Split-level files are:

- `features.jsonl`: fixed material evidence used by train-only calibration.
- `predictions.json`: all successful scene predictions.
- `run_summary.json`: failures and data coverage.
- `metrics.json`: ADE, ALDE, APE, MnRE, RMSE, and 90% interval coverage.

## Evaluation Protocol

For ground-truth mass \(m\) and prediction \(\hat m\), this code reports the
same principal metrics used by NeRF2Physics and PUGS:

\[
\mathrm{ADE}=|m-\hat m|,\quad
\mathrm{ALDE}=|\log m-\log\hat m|,\quad
\mathrm{APE}=|m-\hat m|/m,
\]

\[
\mathrm{MnRE}=\min(m/\hat m,\hat m/m).
\]

MnRE is higher-is-better; the other errors are lower-is-better. The repository
also reports 90% posterior interval coverage to expose overconfidence.

### Paper comparison and figures

Convert an external baseline first when it stores scalar predictions or PUGS-style
`[low, high]` intervals in a JSON object:

```bash
tactile3d import-mass-baseline \
  --predictions results/pugs_predictions.json \
  --targets data/abo_500/filtered_product_weights.json \
  --method PUGS \
  --output outputs/baselines/pugs_normalized.json
```

Then evaluate every method on the exact intersection of scene IDs and generate
the paper figures:

```bash
tactile3d paper-figures \
  --run NeRF2Physics=outputs/baselines/nerf2physics_normalized.json \
  --run PUGS=outputs/baselines/pugs_normalized.json \
  --run PartPhy3D=outputs/abo500_mesh/test/predictions.json \
  --bootstrap-samples 2000 \
  --output outputs/paper/comparison
```

The command exports editable SVG/PDF, 600 dpi PNG/TIFF, a source-data CSV,
paired bootstrap 95% confidence intervals, a predicted-versus-ground-truth
plot, per-object error distributions, and uncertainty calibration. For PUGS
interval predictions the point estimate is the interval midpoint, matching its
released evaluation code; for PartPhy3D it is the posterior mean.

Generate the part segmentation, effective density, Young's modulus, mass
posterior, and interior-structure panels for an individual object:

```bash
tactile3d object-figures \
  --prediction outputs/abo500_mesh/test/scenes/SCENE_ID/prediction.json \
  --mesh outputs/abo500_mesh/test/scenes/SCENE_ID/mesh_metric.ply \
  --face-labels outputs/abo500_mesh/test/scenes/SCENE_ID/face_labels.npy \
  --output outputs/paper/objects/SCENE_ID
```

The full reporting protocol and interpretation of every metric are in
[`docs/evaluation_protocol_zh.md`](docs/evaluation_protocol_zh.md).

## Baseline Relationship

- [NeRF2Physics](https://github.com/ajzhai/NeRF2Physics) reconstructs a NeRF,
  fuses language-aligned point features, and integrates inferred density.
- [PUGS](https://github.com/EverNorif/PUGS) uses geometry- and region-aware 3D
  Gaussians and a Gaussian-specific volume estimator.
- This repository replaces the neural scene representation with an explicit,
  repairable mesh. Its OpenCLIP fusion and integral interfaces follow those
  baselines, while hidden-interior hypotheses and shell/core integration are
  mesh-specific additions. It does not claim to reproduce either paper's
  published numbers.

## Limitations

- A watertight mesh can still contain the wrong internal structure. A hollow
  product and a solid product may have identical visible surfaces.
- ABO-500 mass supervision is object-level. Per-part density and modulus require
  a separately measured material/tactile dataset for quantitative evaluation.
- Mesh and camera poses must be in the same coordinate system for projected
  part crops to be meaningful.
- OpenCLIP estimates appearance-based material evidence; it cannot distinguish
  visually identical materials with different composition.
- This is visual physical-property inference. A claim of tactile perception
  requires contact observations such as force-displacement or GelSight/DIGIT.

## Legacy Demo

The original self-contained single-image geometry smoke test remains available:

```bash
bash scripts/run_demo.sh
```

## License

Project code is MIT licensed. ABO-500, OpenCLIP model weights, and all external
baselines retain their own licenses and must be reviewed separately.

BibTeX entries for the research baselines are in `docs/references.bib`.

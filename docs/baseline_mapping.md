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

### Published reference scores (verified 2026-11-12)

| Published method | ADE kg ↓ | ALDE ↓ | APE % ↓ | MnRE ↑ |
| --- | ---: | ---: | ---: | ---: |
| NeRF2Physics | 12.725 | 0.736 | 104.0 | 0.564 |
| PUGS | 9.461 | 0.661 | 76.7 | 0.576 |
| VoMP, as reported by Gen2Physics | 8.433 | 0.631 | 88.7 | 0.576 |
| Gen2Physics | 8.447 | 0.651 | 117.0 | 0.600 |

Sources: [PUGS Table I](https://arxiv.org/html/2502.12231v1#S4.T1),
[Gen2Physics Table 1](https://arxiv.org/html/2608.23869v1).
APE is converted from fraction to percent. These are published reference numbers,
not our reruns. PUGS explicitly reports 500 objects in its ablations; our cached
NeRF2Physics rerun uses the released 100-object test split. Those cohorts cannot
be ranked together as a paired comparison. Gen2Physics reports all methods on
ground-truth ABO assets; its exact scene list and implementation must be locked
before a matched rerun.

For a fair new comparison, freeze scene IDs, metric scale and GT geometry track,
material priors, VLM backbone and query/image/token budgets; report end-to-end
reconstruction separately. Material segmentation needs material annotations;
part names do not replace them. Compare mass by all four metrics, and compare
QEM by achieved face count, geometry distance, mass/COM/inertia retention and
edit consistency. Modulus, friction and hardness require suitable labels and
contact/test conditions; colored prior maps alone do not measure their accuracy.

Gen2Physics already combines multi-view segmentation, VLM refinement and
solid/hollow reasoning. Our contribution candidate is the geometry-constrained
construction state shared by physical integration, simplification and editing,
with uncertainty calibration still to be demonstrated across objects.

### Shared methodological foundations

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


### VLM construction and component-support integration (2026-11-12)

The earlier geometric-cluster driver is preserved at
[commit 1a0559f](https://github.com/fanyizhi/PartPhy3D/blob/1a0559f/src/tactile3d/experiments/vlm_mass.py).
Its unconstrained five-object pilot obtained ADE 16.4381 kg; replaying the same
answers with library thickness means gave 12.7764 kg. The 20 mm cloth answer
violates the supplied 2–5 mm bounds. Those diagnostic predictions are not
selected as the main method; their raw records remain in the audit.
The finite-cell integration API now lives in `tactile3d.physics.fields`.

The current driver uses the existing VLM/geometry component state instead of
eight geometric clusters. Closed wood boundaries are used directly; other
components use explicitly recorded convex-completion hypotheses. A planar
underside requires a panel-thickness or visual-cap hypothesis. Cloth thickness
and material densities remain the same fixed library means as the controls.
Three independently worded questions receive four actual selected views and
two numbered component diagrams; previous answers are not fed into later calls.
The equal mean of their mass predictions is not a calibrated posterior.

Structural domains have precedence over upholstery. If E_i is a completed
envelope, its assigned domain is D_i = E_i minus all higher-priority envelopes.
The core is D_i intersected with the inward-offset original envelope; the shell
is the remainder. Cropping an overlap therefore creates no artificial cloth
coat. Domain unions conserve volume, and density-weighted zeroth, first and
second moments supply mass, center of mass and inertia. Hidden phases use a
homogenized density; their exact internal positions remain unresolved.

On the exposed chaise case, the three predictions are 41.9033, 49.1507 and
47.2424 kg; the mean is **46.0988 kg**, against catalogue mass **29.0299 kg**.
This still overestimates absolute mass. All three answers propose a 15 mm bottom
panel. Exact repartitioning with a visual-cap alternative and a separate
hidden-wood-as-air ablation quantify the sensitivity of these assumptions.
Neither counterfactual is selected using the target mass. Predictions were
sealed before the new run opened the catalogue label.

The same frozen hypotheses now support a whole-body QEM comparison. Mean
physical errors below measure retention of the original conditional domain
model, rather than error against measured physical truth. Global QEM is rerun
at each paired achieved triangle count. Both component-preserving controls and
all three requested budgets are included in the result record.

| Requested budget | Method | Actual faces | Surface P95 mm | Mass error % | COM error mm | Inertia error % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2400 | Domain-density QEM | 2397 | 4.6349 | 1.2403 | 1.5856 | 1.3223 |
| 2400 | Matched global QEM | 2396 | 175.2563 | 19.3874 | 35.9575 | 16.2744 |
| 1200 | Domain-density QEM | 1215 | 18.6214 | 6.5781 | 4.2877 | 4.8380 |
| 1200 | Matched global QEM | 1215 | 173.0117 | 24.4066 | 43.3888 | 24.2413 |
| 600 | Domain-density QEM | 791 | 36.1865 | 1.6317 | 50.6582 | 16.8633 |
| 600 | Matched global QEM | 791 | 43.7530 | 38.1771 | 31.3705 | 40.4016 |

At the lowest budget, the domain-density allocation has worse COM retention
than the global control. QEM can stop above the requested count, so plots use
achieved counts. Disappeared or dimension-collapsed solid supports contribute
zero material volume and are recorded; they are never replaced by a guessed
underside-panel thickness. Missing reference components are rejected during
inference. Local density edits preserve the domains and obey the analytic
mass change V_i times delta-rho.

```bash
python -m tactile3d.experiments.vlm_mass \
  --root /path/to/abo_500 --reference outputs/reference \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --output outputs/component_construction
python scripts/run_reference_mesh_study.py domain-evaluate \
  --output outputs/reference --construction outputs/component_construction \
  --figures outputs/component_construction/figures
```

The reference directory comes from the existing prepare/infer/evaluate stages,
including its geometry-constrained part state and numbered evidence sheets.
`domain-figures` regenerates only the figure from saved, hash-checked reduced
meshes. [Rendered panels and source data](../paper_assets/chaise_reference_20261111_final/README.md)
and [complete development records](results/mass_integration_audit_20261112.json)
separate absolute mass prediction from conditional simplification retention.
The complete PUGS/Gen2Physics cohort comparison and construction-probability
calibration remain unfinished. The next modeling target is a geometry-supported
hidden-frame hypothesis instead of unconstrained hidden-wood volume fractions.

### Geometry-feedback refinement and matched-query controls

The next development iteration freezes the prior foam fractions, bottom panel,
surface materials, densities, metric mesh and six input images. Three follow-up
queries audit only additional hidden wood. All reported predictions average all
three trials, with no best-trial selection and no mass label in the prompt.

| Same exposed chaise | Total queries including shared prior | Mean mass kg | Absolute error kg | APE % |
| --- | ---: | ---: | ---: | ---: |
| Previous independent-question model | 3 | 46.0988 | 17.0689 | 58.7976 |
| Follow-up without numerical geometry | 6 | 44.9486 | 15.9187 | 54.8357 |
| Follow-up with numerical geometry, free fraction | 6 | 30.5182 | 1.4883 | 5.1267 |
| Follow-up with numerical geometry, beam inventory | 6 | 32.4678 | 3.4379 | 11.8428 |

Catalogue target: 29.029888 kg. Follow-up arms use the same seeds, image budgets
and maximum output-token budgets; actual output lengths and runtimes are recorded
separately. The no-numerical-geometry control removes role volumes, diagonals and
section dimensions from the prompt, with the corresponding wording adjusted.
It retains images and role names. This is one previously exposed development
object, not a held-out benchmark or calibrated hidden-structure posterior.

The beam inventory computes V_wood = sum(count * length * section_area), using
oriented-bounding-box cross-section proxies of visible rails. Invalid dimensions
or combined foam/wood volumes are rejected, not clipped. Beam locations, overlap
and attachment geometry remain unresolved, so this is a volume prior, not recovered
internal CAD. The beam arm is worse than the free-fraction refinement here and
is retained as an ablation. Geometry-feedback fraction refinement is the default
candidate for the next multi-object evaluation.

Historical same-object outputs are NeRF2Physics released-cache 20.5979 kg and
PUGS-Qwen 4.5383 kg. These use different representations, priors and query budgets;
they are not a controlled ranking. Official PUGS GPT-4o and Gen2Physics reruns
remain unfinished. Their published cohort scores above are separate references.

The refined state also completed all three matched-QEM budgets. At **2398 faces
for both methods**, domain-density QEM has mean mass/COM/inertia retention errors
**0.8672%, 1.2287 mm, 0.8745%**, versus global QEM **30.1332%, 52.6984 mm,
23.8388%**. At the lowest budget, COM retention remains weaker than global QEM.
The current `domain_retention` figure now shows this refined state; the earlier
state's full results remain in the audit. Retention errors still concern the
same conditional model rather than measured COM or inertia.

```bash
python -m tactile3d.experiments.vlm_mass \
  --root /path/to/abo_500 --reference outputs/reference \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --refine-from outputs/component_construction \
  --refine-representation fractions --output outputs/refined_construction
```

Use `beams` or `fractions-no-geometry` for the two controls. The shared prior
prediction seal and image hashes are checked before inference. All prediction
files are sealed before the runner opens the catalogue target. The next required
evidence is cross-object generalization, followed by train/validation-only
probability calibration and a locked matched-cohort comparison.

### Cross-object development protocol

`configs/train_furniture3_development.json` locks the first three training IDs
whose existing captions match the declared upholstered-furniture keywords, in
the released order, without reading mass labels or replacing objects. This is
development data, not the held-out test cohort. The original ASIN-normalized
train/validation/test lists have no overlaps.

The reference entry now accepts `--scene`; it no longer requires every new
asset to have the chaise's 15 native candidates. New scenes never receive the
historical manually reviewed chaise role file. Four calibrated views and two
numbered diagrams are generated from each scene's own cameras and geometry.

For fragmented meshes, the new evidence hierarchy keeps large native candidates
separate and groups small candidates by coarse texture RGB. In the locked
cohort, 299, 382 and 49 native candidates yield 9, 7 and 20 evidence groups.
This compression is only for VLM evidence. Every native ID is retained exactly
once, physical integration remains over native domains, and no parent convex
hull is created. Group-inherited materials and roles remain hypotheses; mixed
or unknown groups require further subdivision. Joint material/role voting does
not establish calibrated correctness.

```bash
python scripts/prepare_ten_objects.py --root /path/to/abo_500 \
  --assets /path/to/abo_500/meshes_reference --output outputs/train_inputs \
  --cohort configs/train_furniture3_development.json
python scripts/run_reference_mesh_study.py prepare --root /path/to/abo_500 \
  --scene B07HSDY9FC_ATVPDKIKX0DER --output outputs/new_reference
python scripts/run_reference_mesh_study.py bind-native --output outputs/new_reference
```

`bind-native --surface-audit` independently asks about outer coatings versus
hidden structural material and saves a separate audit state; it never replaces
the frozen initial state. Added leg preflight checks reject bindings inconsistent
with z-up ground support and vertical extent before physical inference. This
rejects impossible bindings without inferring material from a semantic role.
Grounding is therefore checked before trusting a closed mesh as a material domain.

The current cross-object audit does not establish generalization of the earlier
chaise improvement; a complete refined-cohort score is not available. Numerical
feedback cannot correct a wrongly bound surface material by itself. Subsequent
work must resolve component-ID/material grounding and concave or hollow support
before calibrating internal occupancy. Detailed unsuccessful-run diagnostics
remain local and are not manuscript figures.

GLB length/version are checked before parsing. Catalogue evaluation resolves
both complete scene IDs and ASIN keys. `vlm_mass --evaluate-only --root ...
--output ...` evaluates an already sealed prediction without another model call.
`--resume` reuses a raw response only when its prompt and frozen input hashes
match; reports and the previous seal are excluded from the new prediction seal.
Objects with no independent solid domain skip the optional solid-density edit
control instead of aborting their mass prediction.

### Local grounding and closed-boundary diagnostic (2026-11-13)

`bind-local` selects at most three evidence groups, prioritizing geometric leg
conflicts and then uncertain groups by area. Each query receives four original
views, an isolated textured target and a blue-highlighted context. The artificial
highlight is explicitly excluded as material evidence. Three joint material/role
votes are saved per target; this agreement is not a calibrated probability.
Original states remain unchanged. Mixed or unknown materials stop integration
with an explicit error instead of receiving a guessed density.

On training object `B075X5TN2K_ATVPDKIKX0DER`, the wide upholstered body was
previously bound as a wooden leg. Local evidence returned fabric/upholstery in
all three trials. A fixed-parameter 2-by-2 diagnostic holds the pre-existing
material density, fabric thickness, foam occupancy (0.90), hidden-wood fraction
(0.07), and underside hypothesis constant:

| Binding | Upholstery boundary | Mass (kg) | APE (%) |
| --- | --- | ---: | ---: |
| Original | Convex completion | 114.0251 | 214.2280 |
| Original | Preserve closed boundary | 113.6537 | 213.2047 |
| Local target audit | Convex completion | 43.8756 | 20.9115 |
| Local target audit | Preserve closed boundary | 36.1838 | 0.2855 |

The catalogue target is 36.28736 kg. This is a single-object mechanism diagnostic,
not the complete VLM method or a test-set benchmark. The complete three-query
construction procedure on the corrected state predicts 28.7431 kg (20.7903%
APE), averaging all three hypotheses. It must not be replaced by the fixed-prior
diagnostic. Exact raw artifacts, predictions and seals are retained on the server
under `outputs/local_binding_geometry_audit_20261113` and
`outputs/local_binding_fullmass_20261113`; the local evidence manifest records
their file hashes.

`--geometry-policy closed` preserves valid concave closed upholstery instead of
filling empty space with its convex hull. Convex erosion uses inward plane
offsets; concave erosion uses an inscribed 16-segment polyhedral ball, which
underestimates the Euclidean shell thickness contribution. This is a stated
geometric approximation, not recovered internal structure. Open inputs retain
explicit convex/prism completion. Geometry and part-state hashes are recorded.

Fraction refinement uses a fixed region-ID dictionary to prevent omitted broad
roles. Numerical feasibility is checked outside the prompt; exposed numerical
bounds remain an explicit diagnostic option. This improves response coverage,
not established physical accuracy. In the two completed training objects,
numeric-geometry refinement did not outperform its no-numeric-geometry control;
on the sectional sofa both returned 39.0969 kg against 41.0047 kg. The first
training chair was not covered by that refinement experiment. There is no complete three-object refined
score, and these development runs do not establish a gain over PUGS or
Gen2Physics. Detailed unsuccessful-run diagnostics remain local.

An additional `--binding-evidence isolated` control uses only the textured target
and metric extent. On the first training chair, the contextual queries repeatedly
described the whole chair as mixed material, even when their explanation called
the target a wooden leg. Isolated evidence returned wood/leg in all three trials;
the two thin trim targets received plastic hypotheses. These are inferred labels,
not independently annotated material accuracy. The additional queries are part
of the actual development cost, not a budget-matched comparison win.

Numerical replay exposed two independent implementation defects. The historical
cached-name resolver mapped an unrecognized `screws` proposal to plastic,
attaching its 8000 kg/m3 density to plastic targets. Native inference now uses
strict name resolution: fasteners map to metal, unknown names remain unknown,
and absent plastic uses the pre-existing material library mean with a recorded
source/hash. Historical generic-pilot APIs retain their explicit legacy default;
new native inference does not use that fallback. Secondly, a watertight,
winding-consistent trim had globally inward orientation. Closed mode now flips
that orientation without moving vertices before deciding whether completion is
needed. Open or inconsistent boundaries are not certified by this operation.

Solid domains also declare that their core equals the entire assigned solid.
This avoids an unnecessary coincident-surface intersection and subtraction that
created signed numerical slivers on the thin trim. No negative volume is clipped.
The same three sealed construction answers were replayed under the four density
and orientation combinations in `outputs/first_chair_numeric_fixes_20261113`.
These fixes reduced the numerical overestimate but did not resolve its full
construction error. Open upholstered geometry completion and hidden occupancy
remain the next development targets; no target-fitted parameter was introduced.

Refinement changes only hidden-wood core density while freezing the partition,
foam, thickness and underside hypothesis. It therefore reuses sealed unit-density
moments exactly instead of recomputing expensive identical CSG. Resume verifies
input hashes, construction mode and raw prompts before reusing responses.

The locked comparison recipe is `configs/mass_comparison_protocol.json`.
`paper-figures` requires `--cohort` and refuses incomplete rankings; outputs
include scene IDs and prediction-file hashes. Report ADE, ALDE, APE and MnRE
together with paired scene bootstrap intervals. Official-model, shared-VLM and
reference-geometry tracks remain separate, with actual query and image budgets
including local audits. Train/validation calibration must precede freezing the
held-out evaluation recipe. Conditional QEM retention remains separate from
absolute mass accuracy.

```bash
python scripts/run_reference_mesh_study.py bind-local \
  --output outputs/new_reference --model /path/to/Qwen3-VL-8B-Instruct
# Optional controlled follow-up for unresolved targets; keep its output separate.
python scripts/run_reference_mesh_study.py bind-local \
  --output outputs/new_reference \
  --part-state outputs/new_reference/part_state_local_audit.json \
  --binding-evidence isolated --model /path/to/Qwen3-VL-8B-Instruct
python -m tactile3d.experiments.vlm_mass \
  --root /path/to/abo_500 --reference outputs/new_reference \
  --part-state outputs/new_reference/part_state_local_audit.json \
  --geometry-policy closed --model /path/to/Qwen3-VL-8B-Instruct \
  --output outputs/local_grounded_construction
```

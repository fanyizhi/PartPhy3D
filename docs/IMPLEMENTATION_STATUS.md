# Implementation status

Version 0.6.4 adds executed five-object generation-stage sparse weight decoding,
catalog dimension extraction for all 500 objects, a 300/50/50/100 image-scale
baseline with independent scale calibration, conditional closed local layers,
material edits and numerical rigid responses. Human annotation and Mesh-projection
evaluation tools are implemented; independent part masks remain missing.
See [current methods](METHODS_V064_zh.md) and
[actual evidence](results/geometry_physics_20261107/README_zh.md).
These do not complete real interior inference, shared latent training,
interface/physics-aware generation budgets or whole-object physical accuracy.

Updated 2026-11-07. Runnable software, executed experiments and validated research
claims are tracked separately.

Version 0.6.3 executes Qwen/SAM2 visible-surface correspondence and schematic
material/conditional-prior edits on five generated Meshes, plus the official
NeRF2Physics released-cache property stage. Unknown surfaces remain unknown;
real closed material regions and full-method mass accuracy are still missing.
See the [complete idea audit](RESEARCH_IDEA_AUDIT_20261107_zh.md) and
[actual outputs and replay](results/surface_physics_20261107/README_zh.md).
The 8B mass-answer QLoRA experiment now covers 300/100/100 objects, rather than
only the earlier two-step integration test: ADE improved but APE worsened.
This does not validate interior-material learning or general physical inference.

Version 0.6.1 includes the fixed five-object subset in the GitHub repository,
unified requirements installation (including the project itself), a read-only
installation/data check, and a code-plus-data server archive builder. Dependency
import validation and CPU tests alone do not establish GPU training compatibility.
Version 0.6.2 adds executed five-object GPU diagnostics and a two-step QLoRA
integration test. See [actual results and limitations](results/server_pilot_20261106/README_zh.md).

| Component | Current evidence | Remaining work |
| --- | --- | --- |
| Mesh-native physical mainline (v0.6.0) | Direct closed-boundary moments, Mesh CSG shells, disjoint regions, replay and material edits run without voxelization | Evidence-backed regions from real images and independent physical measurements |
| Mesh-basis response learning | Four CSG candidates, actual CPU training, sealed synthetic test predictions | Pixel/Mesh encoders, real edit references, Qwen training |
| Part states and SI units | Implemented and tested | Real multimodal representation and richer constructions |
| Density, mass, COM and full inertia | Analytic synthetic checks pass | Independent real-object accuracy |
| Material, closed-box and donor editing | Commands, rollback, snapshots and replay run | Arbitrary poses, free-form layers and mechanics |
| Material-color preview | One reference ABO asset rendered | Learned selection and physical-state binding |
| Bound region material editing | Known-design multiview colors and physical fields synchronize; analytic and replay checks | Checked real segmentation-to-volume correspondence and independent measurements |
| Ten-object rule baseline | Executed on reference geometry | Real models and held-out evaluation |
| Fixed-cohort comparison | Implemented; missing methods have no scores | Actual published baseline reproduction |
| InstantMesh and Qwen path | Five real observed-view reconstructions; 2B/8B one/three-view mass baselines, sealed predictions and logs | Fine part correspondence, metric scale and learned physics; adapter is not the official single-image pipeline |
| QLoRA | Two actual optimization steps, answer-only loss, separate saved adapter and unchanged 2B base weights | Credible real material supervision and independent generalization evaluation |
| Mesh-camera coordinates | Explicit glTF-to-camera rigid rotation; five reference assets / 15 view audit | Per-dataset scale verification and generated geometry quality |
| Shared-state response learning | CPU NumPy training and held-out synthetic evaluation run | Real pixel/part encoders, independent edited references and strong controls |
| Part refinement and multimodal fusion | Research target | Labels, training and controlled ablations |
| Final surface construction | Conforming field-projected refinement runs on analytic field | Global accuracy constraints and real decoder integration |
| Non-watertight ABO input | Strict failures retained; explicit voxel-envelope baseline runs | Geometry registration, envelope convergence and material accuracy |
| QEM/global physical proxies | Implemented baseline | Local interface/physics-aware construction |
| Robot/deformable interaction | Not executed | Independent reference responses and fixed test conditions |

Acceptance proceeds from mathematical correctness to real model execution,
controlled mechanisms, independent held-out accuracy and controlled interaction.
No stage is completed by silently substituting hand rules or inspecting an animation.

Version 0.5.0 provides `scripts/reproduce_all.py` and audited source/wheel/sdist
packaging. Read [the complete guide](METHODS_AND_REPRODUCTION_zh.md) and
[actual results](results/reproduction_v0.5.0_zh.md). The two-object real rule
baseline has large errors, and task-only refinement loses global volume accuracy.

Version 0.5.1 adds [bound material editing](material_edit_v1_zh.md). Physical state
schema 2 retains optional material elasticity and display colors while reading
schema 1. Legacy standalone color preview is separate from the new bound asset.

Version 0.6.0 adds `scripts/reproduce_mesh_native.py` as the preferred mathematical
and software acceptance path. See [its scope and limitations](mesh_native_v0.6_zh.md).
`scripts/reproduce_all.py` still runs the historical voxel-based diagnostic suite.
Neither entry point silently replaces real reconstruction, segmentation or VLM training.

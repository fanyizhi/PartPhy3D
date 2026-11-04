# Implementation status

Updated 2026-11-04. Runnable software, executed experiments and validated research
claims are tracked separately.

| Component | Current evidence | Remaining work |
| --- | --- | --- |
| Part states and SI units | Implemented and tested | Real multimodal representation and richer constructions |
| Density, mass, COM and full inertia | Analytic synthetic checks pass | Independent real-object accuracy |
| Material, closed-box and donor editing | Commands, rollback, snapshots and replay run | Arbitrary poses, free-form layers and mechanics |
| Material-color preview | One reference ABO asset rendered | Learned selection and physical-state binding |
| Ten-object rule baseline | Executed on reference geometry | Real models and held-out evaluation |
| Fixed-cohort comparison | Implemented; missing methods have no scores | Actual published baseline reproduction |
| InstantMesh and Qwen path | External command adapter and HTTP backend exist; real weights not executed | GPU compatibility, inference, correspondence and logs |
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

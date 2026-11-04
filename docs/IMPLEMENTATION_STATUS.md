# Implementation status

Updated 2026-11-04. Runnable software, executed experiments and validated research
claims are tracked separately.

| Component | Current evidence | Remaining work |
| --- | --- | --- |
| Part states and SI units | Implemented and tested | Learned representation and richer constructions |
| Density, mass, COM and full inertia | Analytic synthetic checks pass | Independent real-object accuracy |
| Material, closed-box and donor editing | Commands, rollback, snapshots and replay run | Arbitrary poses, free-form layers and mechanics |
| Material-color preview | One reference ABO asset rendered | Learned selection and physical-state binding |
| Ten-object rule baseline | Executed on reference geometry | Real models and held-out evaluation |
| Fixed-cohort comparison | Implemented; missing methods have no scores | Actual published baseline reproduction |
| InstantMesh and Qwen path | Planned, not executed in current evidence | Weights, inference, correspondence and logs |
| Part refinement and multimodal fusion | Research target | Labels, training and controlled ablations |
| Generation-stage adaptive construction | Research target | Controllable extraction, transitions and measured cost |
| QEM/global physical proxies | Implemented baseline | Local interface/physics-aware construction |
| Robot/deformable interaction | Not executed | Independent reference responses and fixed test conditions |

Acceptance proceeds from mathematical correctness to real model execution,
controlled mechanisms, independent held-out accuracy and controlled interaction.
No stage is completed by silently substituting hand rules or inspecting an animation.

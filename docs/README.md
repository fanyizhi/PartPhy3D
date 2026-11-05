# Documentation

Start with the installation guide and implementation status. Dated notes preserve
research decisions; they are not evidence that all proposed modules are complete.

| Need | Read |
| --- | --- |
| Run the Mesh-native, non-voxel physical mainline | [v0.6.0 implementation and reproduction, Chinese](mesh_native_v0.6_zh.md) |
| Inspect new independent references and synthetic results | [v0.6.0 executed results](results/mesh_native_v0.6_zh.md) |
| Follow every implemented step, loss and command | [Complete v0.5.0 guide, Chinese](METHODS_AND_REPRODUCTION_zh.md) |
| Install and run on CPU | [Quick start, Chinese](QUICKSTART_zh.md) |
| Prepare ABO images, cameras and meshes | [Data](DATA.md) |
| Reproduce supported experiments | [Experiment commands](EXPERIMENTS.md) |
| Check what actually works | [Implementation status](IMPLEMENTATION_STATUS.md) |
| Inspect recorded numerical evidence | [Results](results/README.md) |
| Understand state, identities and interfaces | [Part state contract](part_state_and_edit_contract_zh.md) |
| Write/replay edits | [Editing commands](edit_commands_2026-11-04_zh.md) |
| Bind material colors and physics to one region | [Material editing v1, Chinese](material_edit_v1_zh.md) |
| Review the planned method step by step | [Method review](method_step_review_2026-11-04_zh.md) |
| Review the complete proposed study and acceptance gates | [Research blueprint, Chinese](research_blueprint_2026-11-04_zh.md) |
| Match each problem to a method and testable advantage | [Problem/solution/evidence, Chinese](problem_solution_advantages_2026-11-04_zh.md) |
| Distinguish generation from simplification | [Generation and preview](generation_and_material_preview_2026-11-04_zh.md) |
| Plan mass and physical comparisons | [Evaluation protocol](evaluation_protocol_zh.md) |
| Plan the next accepted milestone | [Roadmap](ROADMAP.md) |

The main package is named `tactile3d`; the repository and research project are
named **PartPhy3D**. This preserves existing command and import compatibility.
SI units are used: meter, kilogram, second, pascal and kilogram per cubic meter.

Historical material in `archive/` may describe an earlier scope. The current
implementation status and versioned state contracts take precedence when using
the code. No document authorizes starting jobs on a shared server.

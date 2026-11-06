# Recorded evidence

2026-11-07 adds [executed real-Mesh surface mapping and material edits, plus
the official NeRF2Physics released-cache property stage](surface_physics_20261107/README_zh.md).
The [idea audit](../RESEARCH_IDEA_AUDIT_20261107_zh.md) distinguishes every
implemented mechanism from missing evidence. PUGS and full-method mass remain
unscored; the mixed 300/100/100 QLoRA result is recorded in
[the mass-training report](abo_mass_sft_20261107/README_zh.md).
Historical status statements below describe their respective experiments.

2026-11-06 adds [real five-object GPU diagnostics](server_pilot_20261106/README_zh.md):
2B/8B view-count mass baselines, observed-view frozen reconstruction, coordinate
audits and a two-step QLoRA integration test. Raw numbers and adverse cases are
retained. These do not establish the full learned physical method or reproduce
published comparison methods.

Version 0.6.0 adds the Mesh-native physical path and Mesh-basis response learning.
See [executed checks, full metrics and limitations](mesh_native_v0.6_zh.md),
[material sections](mesh_native_v0.6_sections.png) and
[local validation record](mesh_native_v0.6_validation.json). These are known-design
and synthetic checks; the newer five-object diagnostics are recorded above.
Published baselines remain pending.

Version 0.5.1 adds a fixed-geometry, fixed-occupancy material edit on a known stool.
The [three views](material_edit_v1_views.png), [physical changes](material_edit_v1_physics.png),
[CSV](material_edit_v1_physics.csv) and [audit report](material_edit_v1_report.json)
come from specified dimensions and material values, not real-object inference.
The total mass changes from 5.120 to 6.432 kg. No pixels outside the selected
visible region change under the fixed simple renderer. See [scope and reproduction](../material_edit_v1_zh.md).

Version 0.5.0 adds executed shared-construction learning, adaptive-surface and
real two-object rule diagnostics. See [the execution report](reproduction_v0.5.0_zh.md).
The bundled figures and CSV preserve adverse results as well as improvements.

The CSV records the already executed fixed ten-object development comparison:
reference ABO geometry, cached language suggestions and untrained construction rules.
It is not the learned image-to-physical-object pipeline.

Published methods and the full learned method remain `pending`, without scores.
The construction prior did not beat the shell baseline on every metric. These
objects have been used for development and are not an untouched test cohort.

`software_validation.json` records local software checks: the initial editing
subset had 39 passing checks; the exact current count is in that JSON file.
The suite includes two translated analytic-inertia references. Synthetic tests do
not establish real-object accuracy. Check the exact commit's
[Linux CI result](https://github.com/fanyizhi/PartPhy3D/actions/workflows/tests.yml)
before calling that revision passed.

The root figure is a specified synthetic construction. Raw product images,
datasets, checkpoints and private server settings are not distributed in the source
release.

`synthetic_edit.csv` is the numerical source for the default editing figures.
Its reference values come from the specified box dimensions and densities.
Near-zero error verifies integration arithmetic for this construction, not
inferred real-world density or hidden structure.

`observability_audit.json` and `observability_audit.csv` record a simplified
analytic assembly with one material phase, void and unchanged remaining mass.
They check observation rank and conditioning under declared interventions, with
known scale and new densities. The noise calculation uses an illustrative assumed
measurement error. These records do not establish RGB-only identifiability,
trained-model performance or real-object accuracy.

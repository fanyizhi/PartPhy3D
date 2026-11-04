# Recorded evidence

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

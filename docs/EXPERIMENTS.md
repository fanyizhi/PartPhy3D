# Experiment commands and acceptance

Commands assume the project root and an editable installation:

```bash
python -m pip install -e ".[dev,figures]"
```

## E0: independently specified editing reference

```bash
python scripts/edit_parts.py --output outputs/edit-demo
python scripts/edit_parts.py \
  --plan examples/wood_to_hollow_steel.json \
  --output outputs/edit-json
python scripts/edit_parts.py \
  --state outputs/edit-demo/state_000 \
  --plan outputs/edit-demo/program.json \
  --output outputs/edit-replay
```

Output directories must be new. The default program includes material, hollow-box
and compatible donor edits. It checks mass, COM and the full inertia tensor
against independent box formulae and checks each incremental update against full
recomputation. The custom JSON/replay path checks recomputation agreement; it does
not automatically acquire an independent physical ground truth.

Inspect `report.json`, `manifest.json`, `physical_updates.csv`, saved states,
PLYs and PNG/SVG/PDF figures. Local validation currently records 86 passing tests:

```bash
python -m ruff check .
python -m pytest -q
```

GitHub Actions runs the CPU tests and both demos on Linux, and stores generated
demo artifacts for 14 days. Check its actual run result; a configured workflow
alone is not a passed check. These tests do not require or start training.

## E1: ten-object reference-mesh development diagnostic

Prepare the data using [DATA.md](DATA.md), then run stages separately:

```bash
python -m tactile3d.experiments.ten_objects \
  --root data/abo_500 --assets data/abo500_ten \
  --config configs/ten_object_pilot.yaml \
  --output outputs/abo500_ten --stage predict
python -m tactile3d.experiments.ten_objects \
  --root data/abo_500 --output outputs/abo500_ten --stage evaluate
python -m tactile3d.experiments.ten_objects \
  --root data/abo_500 --output outputs/abo500_ten --stage plot
python scripts/compare_experiments.py --pilot outputs/abo500_ten
```

The first stage seals predictions before evaluation. Use the same frozen IDs and
retain failed/missing methods. The archived results in `docs/results/` came from
untrained construction rules on reference geometry. These objects have been seen
during development. This experiment is not evidence that the complete learned
method or published baselines have run.

## E2–E5: research targets requiring new evidence

| Experiment | Required evidence | Principal comparisons |
| --- | --- | --- |
| Original physical estimation | Actual generated meshes, independent labels, frozen test set | ADE/ALDE/APE/MnRE; uncertainty calibration only with valid probabilistic predictions |
| Part/structure mechanism | Part and interface references, known constructions | Boundary accuracy, interface quality, construction ablations and physical errors |
| Adaptive final construction | Access to construction/extraction budgets, comparable outputs | Dense vs uniform vs adaptive; geometry and physical error vs actual time/memory |
| Physical editing | Independently specified or measured edited object | Edited mass/COM/inertia error, local vs full update accuracy and elapsed time |

NeRF2Physics, PUGS, Gen2Physics, PhysVGGT and VoMP have differing inputs and
targets. Register only executed, compatible comparisons with recorded upstream
versions and supervision. Do not put publication-level aggregate scores into a
table of per-object reproductions. Pending entries have no numerical scores.

Training launchers and data-preparation modules are research utilities; they are
not evidence of a completed training run. There is no distributed full-method
checkpoint. See [ROADMAP.md](ROADMAP.md) for acceptance and revision criteria.

# PartPhy3D

Part-level physical inference, task-aware mesh simplification and editing with
multi-view VLM evidence and geometric constraints.

[Latest figures](paper_assets/chaise_reference_20261111_final/README.md) ·
[Results](docs/results/chaise_reference_20261111_zh.md) ·
[Reproduction](docs/REFERENCE_MESH_STUDY_zh.md) ·
[Mass baselines](docs/baseline_mapping.md) ·
[Server setup](docs/SERVER_SETUP_zh.md) · [Data](docs/DATA.md) · [中文](README_zh.md)

## Run

```bash
python -m pip install -e ".[study]"
python scripts/run_reference_mesh_study.py prepare --output outputs/reference
python scripts/run_reference_mesh_study.py infer --output outputs/reference
python scripts/run_reference_mesh_study.py evaluate --output outputs/reference
python scripts/run_reference_mesh_study.py figures --output outputs/reference
```

VLM inference additionally requires a GPU, local model weights and training
dependencies; see the setup guide. Use `all` for the complete workflow or
`--resume` to reuse completed stages. Figures are written inside the run directory.

## Layout

- `src/tactile3d/`: inference, geometry, physics and editing modules.
- `scripts/run_reference_mesh_study.py`: the current experiment entry point.
- `configs/`, `tests/`: configuration and verification.
- `paper_assets/chaise_reference_20261111_final/`: current figures and source data.
- `docs/`: methods, data preparation and historical text records.

Dataset images, reference models, checkpoints and intermediate exports are local
resources, not bundled in Git. Current figures describe a reference-geometry
development study; captions identify parameter priors and the four-leg domain.

Code is MIT licensed. Data and third-party components retain their own licenses;
see [THIRD_PARTY.md](THIRD_PARTY.md).

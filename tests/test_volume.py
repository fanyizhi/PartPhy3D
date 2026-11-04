import pytest
import trimesh

from tactile3d.mesh.volume import analyze_mesh_volume


def test_metric_mesh_volume_and_uniform_mass(tmp_path):
    mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    path = tmp_path / "box.ply"
    mesh.export(path)
    result = analyze_mesh_volume(
        path,
        0.01,
        density_kg_m3=1000.0,
        fill_fraction=0.5,
        output_path=tmp_path / "volume.json",
    )
    assert result["volume"]["is_exact_enclosed_volume"]
    assert result["volume"]["m3"] == pytest.approx(24e-6)
    assert result["occupancy"]["occupied_volume_m3"] == pytest.approx(12e-6)
    assert result["uniform_material_mass"]["mass_kg"] == pytest.approx(0.012)


def test_open_mesh_does_not_report_exact_volume(tmp_path):
    mesh = trimesh.creation.box()
    mesh.update_faces([False] + [True] * (len(mesh.faces) - 1))
    path = tmp_path / "open.ply"
    mesh.export(path)
    result = analyze_mesh_volume(path, 1.0, repair_strategy="none")
    assert not result["volume"]["is_exact_enclosed_volume"]
    assert result["volume"]["m3"] is None

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from tactile3d.mesh.repair import repair_mesh, voxel_envelope
from tactile3d.mesh.volume import analyze_mesh_volume


def test_voxel_envelope_has_declared_error_on_known_box() -> None:
    mesh = trimesh.creation.box(extents=[1.0, 0.6, 0.4])
    result, report = voxel_envelope(mesh, 32)
    assert result.is_volume
    assert report["true_internal_geometry_recovered"] is False
    assert result.volume == pytest.approx(mesh.volume, rel=0.25)


def test_strategies_never_silently_substitute_unknown_repair() -> None:
    mesh = trimesh.creation.box()
    with pytest.raises(ValueError, match="Unknown repair"):
        repair_mesh(mesh, "typo")
    with pytest.raises(ValueError, match="分辨率"):
        voxel_envelope(mesh, 999)


def test_closed_envelope_is_not_reported_as_exact_input_volume(tmp_path: Path) -> None:
    file = tmp_path / "box.ply"
    trimesh.creation.box().export(file)
    report = analyze_mesh_volume(file, 1.0, repair_strategy="voxel_envelope")
    assert report["volume"]["is_valid_closed_mesh"]
    assert not report["volume"]["is_exact_enclosed_volume"]
    assert report["volume"]["method"] == "voxel_envelope_proxy"
    assert report["volume"]["m3"] > 0

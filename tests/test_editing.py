import numpy as np
import trimesh

from tactile3d.mesh.editing import replace_mesh_part


def test_part_replacement_recomputes_mass_and_exports_assembly(tmp_path):
    base = trimesh.creation.box(extents=[1.0, 1.0, 0.2])
    base.apply_translation([0.0, 0.0, 0.1])
    top = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
    top.apply_translation([0.0, 0.0, 0.4])
    source = trimesh.util.concatenate([base, top])
    source_path = tmp_path / "source.ply"
    source.export(source_path)
    labels = np.concatenate(
        [np.zeros(len(base.faces), dtype=np.int64), np.ones(len(top.faces), dtype=np.int64)]
    )
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)
    donor = trimesh.creation.icosphere(subdivisions=1, radius=0.25)
    donor_path = tmp_path / "donor.ply"
    donor.export(donor_path)

    report = replace_mesh_part(
        source_path,
        labels_path,
        donor_path,
        1,
        tmp_path / "edit",
        mesh_unit_to_meter=1.0,
        source_density_kg_m3=500.0,
        donor_density_kg_m3=7800.0,
        use_icp=False,
    )
    assert report["physics"]["mass_after_kg"] > report["physics"]["mass_before_kg"]
    assert np.asarray(report["physics"]["inertia_tensor_after_kg_m2"]).shape == (3, 3)
    assert all(value >= 0 for value in report["physics"]["principal_inertia_after_kg_m2"])
    assert report["physics"]["rigid_body_consistency"]["all_passed"]
    assert report["geometry"]["edited_watertight"]
    assert (tmp_path / "edit" / "edited_assembly.ply").exists()
    assert (tmp_path / "edit" / "edit_report.json").exists()

import trimesh

from tactile3d.mesh.physics_lod import LODConstraints, select_physics_constrained_lod


def test_physics_lod_reports_cross_resolution_errors():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    reduced, report = select_physics_constrained_lod(
        mesh,
        target_faces=(900, 500),
        constraints=LODConstraints(
            volume_relative=0.08,
            center_of_mass_relative=0.02,
            inertia_relative=0.1,
            surface_area_relative=0.05,
        ),
    )
    assert len(reduced.faces) < len(mesh.faces)
    assert report["selected"]["passes_constraints"]
    assert 0 < report["face_reduction_fraction"] < 1
    assert report["mass_property_semantics"] == "exact enclosed mass properties"

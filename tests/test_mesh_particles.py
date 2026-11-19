import numpy as np
import pytest
import trimesh

from tactile3d.physics.materials import MaterialPrior, ScalarPrior
from tactile3d.physics.mesh_particles import Construction, sample_mesh_particles
from tactile3d.physics.mpm_validation import coarsen_material_points, run_mpm


def test_solid_route_rejects_ambiguous_and_conflicting_evidence():
    solid = Construction.from_evidence(surface="wood", solid=True)
    assert solid.surface == solid.core == "wood" and solid.thickness_m == 0
    with pytest.raises(ValueError, match="contradictory"):
        Construction.from_evidence(surface="wood", solid=True, core="foam")
    with pytest.raises(ValueError, match="missing"):
        Construction.from_evidence(surface="wood", solid=False)


def test_thin_shell_and_core_particles_conserve_volume_through_mpm_export():
    pytest.importorskip("igl")
    shell = Construction.from_evidence(
        surface="fabric", solid=False, core="foam", thickness_m=0.02
    )
    priors = [
        MaterialPrior(
            name, ScalarPrior(rho, 0, rho, rho), ScalarPrior(1e5, 0, 1e5, 1e5)
        )
        for name, rho in (("fabric", 350), ("foam", 65))
    ]
    points = sample_mesh_particles(
        trimesh.creation.box((0.2,) * 3),
        shell,
        priors,
        finest_m=0.004,
        bulk_m=0.025,
        poisson={"fabric": 0.30, "foam": 0.25},
    )
    depth = 0.1 - np.abs(points["position_m"]).max(axis=1)
    assert np.all(depth > 0)
    np.testing.assert_array_equal(points["material_id"], depth > 0.02)
    volume = np.bincount(points["material_id"], weights=points["volume_m3"])
    np.testing.assert_allclose(volume, [0.2**3 - 0.16**3, 0.16**3], rtol=0.08)
    export = coarsen_material_points(points, 0.04)
    for key in ("mass_kg", "volume_m3"):
        np.testing.assert_allclose(
            np.bincount(export["material_id"], weights=export[key]),
            np.bincount(points["material_id"], weights=points[key]),
        )
    fall = run_mpm(export, dx=0.06, steps=8)
    loaded = run_mpm(export, dx=0.06, steps=8, strain_rate=1)
    assert fall["max_deviation_from_ballistic_m"] < 1e-12
    assert loaded["elastic_energy_j"] > 0 and loaded["jacobian"].min() > 0

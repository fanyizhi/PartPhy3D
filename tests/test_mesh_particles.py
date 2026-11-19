import numpy as np
import pytest
import trimesh

from tactile3d.physics.materials import MaterialPrior, ScalarPrior
from tactile3d.physics.mesh_particles import (
    Construction,
    InteriorDomain,
    sample_disjoint_material_domains,
    sample_mesh_particles,
)
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


def test_generated_void_and_visual_wall_thickness_emit_no_core_points():
    pytest.importorskip("igl")
    with pytest.raises(ValueError, match="Unresolved"):
        Construction.from_visual_hypothesis("plastic", "unknown")
    outer = trimesh.creation.box((0.2,) * 3)
    inner = trimesh.creation.box((0.16,) * 3)
    priors = [
        MaterialPrior(
            "plastic", ScalarPrior(1000, 0, 1000, 1000), ScalarPrior(1e5, 0, 1e5, 1e5)
        )
    ]
    settings = dict(finest_m=0.001, bulk_m=0.020, poisson={"plastic": 0.3})
    generated = sample_mesh_particles(
        outer,
        Construction.from_generated_partition("plastic"),
        priors,
        inner_mesh=inner,
        **settings,
    )
    visual = sample_mesh_particles(
        outer,
        Construction.from_visual_hypothesis("plastic", "hollow", thickness_m=0.02),
        priors,
        **settings,
    )
    for data in (generated, visual):
        assert set(data["material_names"]) == {"plastic"}
        assert np.all(np.abs(data["position_m"]).max(axis=1) >= 0.08)
        assert data["mass_kg"].sum() == pytest.approx(
            1000 * (0.2**3 - 0.16**3), rel=0.05
        )
    assert generated["mass_kg"].sum() == pytest.approx(
        1000 * (0.2**3 - 0.16**3), rel=1e-10
    )


def test_disjoint_material_domains_conserve_mass_and_reject_overlap():
    pytest.importorskip("igl")
    outer = trimesh.creation.box((0.2,) * 3)
    air = trimesh.creation.box((0.07,) * 3)
    fill = air.copy()
    air.apply_translation((0.04, 0, 0))
    fill.apply_translation((-0.04, 0, 0))
    priors = [
        MaterialPrior(name, ScalarPrior(d, 0, d, d), ScalarPrior(1e5, 0, 1e5, 1e5))
        for name, d in (("plastic", 1000), ("foam", 60))
    ]
    settings = dict(finest_m=0.004, bulk_m=0.02, poisson={"plastic": 0.3, "foam": 0.25})
    data = sample_disjoint_material_domains(
        outer,
        "plastic",
        (InteriorDomain(air, "void"), InteriorDomain(fill, "foam")),
        priors,
        **settings,
    )
    assert set(data["material_names"]) == {"plastic", "foam"}
    assert not np.any(
        np.all(
            (data["position_m"] > air.bounds[0]) & (data["position_m"] < air.bounds[1]),
            axis=1,
        )
    )
    mass = 1000 * (outer.volume - air.volume - fill.volume) + 60 * fill.volume
    assert data["mass_kg"].sum() == pytest.approx(mass, rel=1e-7)
    with pytest.raises(ValueError, match="overlap"):
        sample_disjoint_material_domains(
            outer,
            "plastic",
            (InteriorDomain(air, "void"), InteriorDomain(air, "foam")),
            priors,
            **settings,
        )

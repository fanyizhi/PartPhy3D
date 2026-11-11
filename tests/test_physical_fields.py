import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from tactile3d.experiments.single_object import (
    digest,
    evaluate,
    predict_fields,
    voxel_partition,
)
from tactile3d.physics.fields import (
    integrate_density_cells,
    isotropic_mixture_bounds,
    material_conditioned_shell,
)


def test_material_and_thickness_stay_coupled_in_thin_shell_integral():
    result = material_conditioned_shell(
        [.2], .2, [.1, .001], np.array([0, 1]),
        {"density": [100., 8000.]}, {"density": [0., 0.]})
    masses = result["shell_volume_samples_m3"] * [100., 8000.]
    np.testing.assert_allclose(masses, [.4, .32])
    assert result["mean"]["density"][0] * .2**3 == pytest.approx(.36)
    # E[rho] E[t] is 22.725 times E[rho*t] for this analytic example.
    decoupled = np.mean([100., 8000.]) * np.mean([.1, .001]) * .2**2
    assert decoupled / masses.mean() == pytest.approx(22.725)


def test_conditional_field_mean_and_variance_match_explicit_sample_enumeration():
    depth = np.array([.1, .15, .2])
    choices = np.array([0, 1, 0, 1])
    thickness = np.array([.01, .08])
    outer, core = np.array([100., 1000., 200., 900.]), np.array([20., 0., 10., 30.])
    result = material_conditioned_shell(depth, .1, thickness, choices,
                                        {"density": outer}, {"density": core})
    shell = np.clip((thickness[choices][None, :] - np.maximum(depth[:, None]-.1, 0))/.1, 0, 1)
    explicit = shell*outer + (1-shell)*core
    np.testing.assert_allclose(result["mean"]["density"], explicit.mean(axis=1))
    np.testing.assert_allclose(result["std"]["density"], explicit.std(axis=1))
    sample_mass = result["shell_volume_samples_m3"]*outer + (
        len(depth)*.1**3-result["shell_volume_samples_m3"])*core
    assert result["mean"]["density"].sum()*.1**3 == pytest.approx(sample_mass.mean())


def test_conditional_shell_respects_cubic_metric_scale():
    arguments = (np.array([0, 1]), {"density": [100., 8000.]}, {"density": [0., 0.]})
    first = material_conditioned_shell([.2], .2, [.1, .001], *arguments)
    scaled = material_conditioned_shell([2.], 2., [1., .01], *arguments)
    np.testing.assert_allclose(scaled["mean"]["density"], first["mean"]["density"])
    np.testing.assert_allclose(scaled["shell_volume_samples_m3"],
                               1000*first["shell_volume_samples_m3"])


def test_finite_cubes_give_analytic_rectangular_solid_and_translation_invariance():
    pitch = .1
    coordinates = np.stack(np.meshgrid(np.arange(2), np.arange(3), np.arange(4),
                                        indexing="ij"), axis=-1).reshape(-1, 3)*pitch
    density = np.full(len(coordinates), 700.)
    result = integrate_density_cells(coordinates, density, pitch)
    mass = 700*.2*.3*.4
    expected = mass/12*np.diag([.3**2+.4**2, .2**2+.4**2, .2**2+.3**2])
    assert result["mass_kg"] == pytest.approx(mass)
    np.testing.assert_allclose(result["inertia_kg_m2"], expected, rtol=1e-12, atol=1e-14)
    moved = integrate_density_cells(coordinates+[1, 2, 3], density, pitch)
    np.testing.assert_allclose(moved["inertia_kg_m2"], expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(moved["center_of_mass_m"],
                               np.asarray(result["center_of_mass_m"])+[1, 2, 3])


def test_mixture_bounds_recover_pure_phase_and_zero_reuss_bound_for_void():
    low, high = isotropic_mixture_bounds([[1, 0], [.5, .5]], [1e9, 0], [.3, 0])
    assert low[0] == pytest.approx(1e9)
    assert high[0] == pytest.approx(1e9)
    assert low[1] == 0
    assert high[1] == pytest.approx(5e8)
    with pytest.raises(ValueError):
        isotropic_mixture_bounds([[.8, .8]], [1e9, 0], [.3, 0])


def test_overlapping_envelopes_have_unique_owners():
    a = trimesh.creation.box(extents=[.2, .2, .2])
    b = a.copy()
    b.apply_translation([.1, 0, 0])
    mesh = trimesh.util.concatenate([a, b])
    centers, ids, audit = voxel_partition(mesh, np.r_[np.zeros(12, int), np.ones(12, int)],
                                          .025, closing=1)
    assert len(np.unique(centers, axis=0)) == len(centers)
    assert audit["overlap_cells"] > 0
    assert set(ids) == {0, 1}
    assert audit["disjoint_after_assignment"]


def test_prior_samples_integrate_to_cell_mass_without_a_target():
    cfg = yaml.safe_load(Path("configs/single_object_chaise.yaml").read_text())
    cfg["samples"] = 128
    centers = np.array([[0., 0., 0.], [.012, 0, 0], [0, .012, 0], [.012, .012, 0]])
    ids = np.array([0, 5, 6, 7])
    fields = predict_fields(centers, ids, .012, cfg)
    assert fields["moments"]["mass_kg"] == pytest.approx(
        fields["part_samples"].sum(axis=0).mean())
    np.testing.assert_allclose(fields["fractions"].sum(axis=1), 1)
    assert fields["fractions"].min() >= 0


def test_target_changes_only_evaluation_and_tampering_is_rejected(tmp_path):
    root, output = tmp_path/"dataset", tmp_path/"output"
    root.mkdir()
    output.mkdir()
    prediction = {"scene_id": "item", "scenarios": {
        "frozen": {"mean": 2., "q05": 1., "q95": 3.}}}
    (output/"prediction.json").write_text(json.dumps(prediction))
    checksum = digest(output/"prediction.json")
    (output/"prediction_seal.json").write_text(json.dumps({"prediction.json": checksum}))
    (root/"filtered_product_weights.json").write_text('{"item": 2.0}')
    first = evaluate(root, output)
    (root/"filtered_product_weights.json").write_text('{"item": 4.0}')
    second = evaluate(root, output)
    assert first[0]["MnRE"] == 1
    assert second[0]["MnRE"] == .5
    assert digest(output/"prediction.json") == checksum
    (output/"prediction.json").write_text(json.dumps({**prediction, "tampered": True}))
    with pytest.raises(ValueError, match="sealed"):
        evaluate(root, output)

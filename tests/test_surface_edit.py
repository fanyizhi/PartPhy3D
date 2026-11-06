"""在真实冻结标签上检查小部件换材与失败原子性。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tactile3d.physics.materials import load_material_priors
from tactile3d.physics.surface_edit import edit_surface_fields

ROOT = Path(__file__).resolve().parents[1]
SCENE = "B013TGFGWS_A1F83G8C2ARO7P"


def test_real_patch_edit_preserves_other_parts_and_inputs() -> None:
    source = ROOT / "examples/observed_surface_demo" / SCENE
    token = json.loads((source / "part_tokens.json").read_text(encoding="utf-8"))
    original_token = json.dumps(token, sort_keys=True)
    with np.load(source / "surface_fields.npz") as stored:
        labels = stored["labels"]
    density = np.where(labels == 0, np.nan, 350.0)
    youngs = np.where(labels == 0, np.nan, 1e8)
    colors = np.full((len(labels), 4), 127, np.uint8)
    material = next(
        p
        for p in load_material_priors(ROOT / "configs/material_priors.yaml")
        if p.name == "steel"
    )
    result = edit_surface_fields(
        labels,
        density,
        youngs,
        colors,
        token,
        "leather_patch",
        material,
        (77, 114, 156, 255),
    )
    assert result.changed.sum() == 356
    np.testing.assert_array_equal(
        result.density_kg_m3[~result.changed], density[~result.changed]
    )
    np.testing.assert_array_equal(
        result.youngs_pa[~result.changed], youngs[~result.changed]
    )
    np.testing.assert_array_equal(
        result.colors_rgba[~result.changed], colors[~result.changed]
    )
    assert np.all(result.density_kg_m3[result.changed] == 7850)
    assert np.all(result.youngs_pa[result.changed] == 2e11)
    assert json.dumps(token, sort_keys=True) == original_token
    assert np.all(colors == 127)
    assert np.all(density[labels > 0] == 350)
    assert result.tokens["closed_material_volumes_established"] is False
    assert result.tokens["parts"][1]["surface_material"] == "steel"


def test_invalid_edit_rejects_without_mutating_inputs() -> None:
    material = load_material_priors(ROOT / "configs/material_priors.yaml")[0]
    token = {"parts": [{"part_id": "body", "part_index": 1}]}
    density, youngs = np.ones(2), np.ones(2)
    colors = np.zeros((2, 4), np.uint8)
    with pytest.raises(ValueError, match="未知编辑目标"):
        edit_surface_fields(
            np.array([0, 1], np.int32),
            density,
            youngs,
            colors,
            token,
            "missing",
            material,
            (0, 0, 0, 255),
        )
    assert token == {"parts": [{"part_id": "body", "part_index": 1}]}
    assert np.all(density == 1) and np.all(youngs == 1) and np.all(colors == 0)

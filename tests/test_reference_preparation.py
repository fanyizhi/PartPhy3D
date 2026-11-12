import hashlib
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image

from tactile3d.experiments import reference_inference, reference_study
from tactile3d.vision.local_vlm import VLMResponse


@pytest.mark.parametrize("mode,image_count", [("global-context", 6), ("isolated", 1)])
def test_binding_evidence_scope_and_state_provenance(
    tmp_path, monkeypatch, mode, image_count
):
    box = trimesh.creation.box()
    labels = np.zeros(len(box.faces), dtype=int)
    monkeypatch.setattr(
        reference_inference,
        "load_prepared",
        lambda _: (box, labels, [box], np.full((len(box.faces), 3), 0.5)),
    )
    monkeypatch.setattr(
        reference_inference,
        "render_mesh",
        lambda *a, **k: np.full((16, 16, 3), 255, dtype=np.uint8),
    )
    calls = []

    class Model:
        def __init__(self, *a, **k):
            self.provenance = {"fixture": True}

        def generate(self, prompt, images, *a, **k):
            calls.append((prompt, len(images)))
            return VLMResponse(
                json.dumps(
                    {"surface_material": "fabric", "functional_role": "upholstery"}
                ),
                0.0,
                1,
                1,
                0,
            )

    monkeypatch.setattr(reference_inference, "LocalVLM", Model)
    (tmp_path / "reference_arrays.npz").write_bytes(b"frozen geometry")
    state = {
        "geometry_sha256": hashlib.sha256(b"frozen geometry").hexdigest(),
        "parts": [
            {"component_id": 0, "surface_material": "mixed", "functional_role": "frame"}
        ],
    }
    path = tmp_path / "part_state_geometry_constrained.json"
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    (tmp_path / "input_manifest.json").write_text(
        json.dumps({"evidence_groups": [[0]]})
    )
    views = tmp_path / "views_contacts"
    views.mkdir()
    (views / "view_selection.json").write_text(
        json.dumps({"4": [{"frame_index": i} for i in range(4)]})
    )
    for i in range(4):
        Image.new("RGB", (16, 16), "red").save(views / f"view_{i:02d}.png")
    reference_inference.refine_local_binding(
        SimpleNamespace(
            output=tmp_path, binding_evidence=mode, model=tmp_path / "model"
        )
    )
    assert len(calls) == 3 and all(count == image_count for _, count in calls)
    if mode == "isolated":
        assert all("No other components are shown" in prompt for prompt, _ in calls)
    assert path.read_bytes() == original
    name = (
        "part_state_isolated_audit.json"
        if mode == "isolated"
        else "part_state_local_audit.json"
    )
    directory = tmp_path / (
        "local_binding_isolated_audit" if mode == "isolated" else "local_binding_audit"
    )
    updated = json.loads((tmp_path / name).read_text())
    assert updated["parts"][0]["surface_material"] == "fabric"
    assert updated["prior_state_sha256"] == hashlib.sha256(original).hexdigest()
    assert (directory / name).read_bytes() == (tmp_path / name).read_bytes()
    seal = json.loads((directory / "seal.json").read_text())
    assert all(
        hashlib.sha256((directory / file).read_bytes()).hexdigest() == digest
        for file, digest in seal.items()
    )


def test_incomplete_download_is_rejected_before_geometry_processing(tmp_path):
    path = tmp_path / "data/meshes_reference/new/model.glb"
    path.parent.mkdir(parents=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 1024))
    args = SimpleNamespace(
        root=tmp_path / "data", output=tmp_path / "run", scene="new", trials=3
    )
    with pytest.raises(ValueError, match="incomplete"):
        reference_study.prepare(args)
    assert not (args.output / "reference_arrays.npz").exists()


def test_new_scene_uses_own_geometry_views_and_never_receives_chaise_review(tmp_path):
    scene = "new_chair"
    root = tmp_path / "data"
    mesh_path = root / "meshes_reference" / scene / "model.glb"
    mesh_path.parent.mkdir(parents=True)
    box = trimesh.creation.box(extents=[0.2] * 3)
    other = box.copy()
    other.apply_translation([1, 0, 0])
    trimesh.util.concatenate([box, other]).export(mesh_path)
    folder = root / "scenes" / scene
    (folder / "images").mkdir(parents=True)
    frames = []
    for i, angle in zip([0, 10, 20, 30], np.arange(4) * np.pi / 2):
        name = f"images/{scene}_{i:02d}.png"
        Image.new("RGBA", (16, 16), "blue").save(folder / name)
        pose = np.eye(4)
        pose[:3, 3] = [np.cos(angle), np.sin(angle), 1]
        frames.append({"file_path": name, "transform_matrix": pose.tolist()})
    (folder / "transforms.json").write_text(json.dumps({"frames": frames}))
    args = SimpleNamespace(root=root, output=tmp_path / "run", scene=scene, trials=3)
    reference_study.prepare(args)
    reference_inference.prepare_views(args)
    manifest = json.loads((args.output / "input_manifest.json").read_text())
    assert manifest["scene"] == scene
    assert len(manifest["components"]) == 2
    assert manifest["evidence_groups"] == [[0], [1]]
    mesh, labels, pieces, _ = reference_study.load_prepared(args.output)
    assert len(pieces) == 2 and set(labels) == {0, 1}
    assert np.isclose(mesh.volume, 0.016)
    assert not (args.output / "views_contacts/role_review.json").exists()

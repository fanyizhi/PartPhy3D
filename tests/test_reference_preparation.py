import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image

from tactile3d.experiments import reference_inference, reference_study


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

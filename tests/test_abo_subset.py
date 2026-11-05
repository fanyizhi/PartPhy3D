from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from tactile3d.abo500.dataset import ABO500Dataset
from tactile3d.abo500.subset import archive_subset, build_subset, verify_subset
from tactile3d.physics.part_io import write_json


def make_source(root: Path) -> tuple[Path, Path, Path, list[str]]:
    source, meshes = root / "source", root / "reference_meshes"
    ids = [f"ITEM{index}_LOCALE" for index in range(5)]
    weights = {f"ITEM{index}": index + 0.5 for index in range(5)}
    for scene_id in ids:
        scene = source / "scenes" / scene_id
        images = scene / "images"
        images.mkdir(parents=True)
        frames = []
        for index in range(2):
            filename = f"images/{index}.png"
            Image.new("RGBA", (8, 8), (120, 90, 60, 255)).save(scene / filename)
            frames.append(
                {"file_path": filename, "transform_matrix": np.eye(4).tolist()}
            )
        write_json(
            scene / "transforms.json",
            {
                "frames": frames,
                "w": 8,
                "h": 8,
                "fl_x": 10,
                "fl_y": 10,
            },
        )
        write_json(scene / "info.json", {"caption": "fixture", "idx_to_caption": "0"})
        (scene / "features").mkdir()
        (scene / "features/unused.cache").write_text("not required")
        mesh = meshes / scene_id
        mesh.mkdir(parents=True)
        trimesh.creation.box().export(mesh / "model.glb")
    write_json(source / "filtered_product_weights.json", weights)
    write_json(source / "splits.json", {"train": [], "val": [], "test": ids})
    cohort = root / "cohort.json"
    write_json(cohort, {"object_ids": ids, "held_out_benchmark": False})
    return source, meshes, cohort, ids


def test_five_real_format_objects_are_copied_and_alias_labels_mapped(
    tmp_path: Path,
) -> None:
    source, meshes, cohort, ids = make_source(tmp_path)
    output = tmp_path / "subset"
    result = build_subset(source, meshes, output, cohort)
    assert result["num_objects"] == 5
    assert result["num_views"] == 10
    assert result["inference_or_training_executed"] is False
    assert result["objects"][0]["source_weight_key"] == "ITEM0"
    assert not (output / "scenes" / ids[0] / "features").exists()
    original = source / "scenes" / ids[0] / "images/0.png"
    copied = output / "scenes" / ids[0] / "images/0.png"
    assert copied.read_bytes() == original.read_bytes()
    dataset = ABO500Dataset(output, output / "meshes_reference", 1)
    assert dataset.scene_ids("test") == ids
    assert dataset.get_scene(ids[0]).mass_kg == 0.5
    assert len(dataset.get_scene(ids[4]).views) == 2
    with pytest.raises(FileExistsError):
        build_subset(source, meshes, output, cohort)


@pytest.mark.parametrize("broken", ["image", "label", "conflicting_label", "camera"])
def test_invalid_fixed_objects_abort_before_output_creation(
    tmp_path: Path, broken: str
) -> None:
    source, meshes, cohort, ids = make_source(tmp_path)
    if broken == "image":
        (source / "scenes" / ids[0] / "images/0.png").unlink()
    elif broken in ("label", "conflicting_label"):
        path = source / "filtered_product_weights.json"
        weights = json.loads(path.read_text())
        if broken == "label":
            del weights["ITEM0"]
        else:
            weights[ids[0]] = 123
        write_json(path, weights)
    else:
        path = source / "scenes" / ids[0] / "transforms.json"
        metadata = json.loads(path.read_text())
        metadata["frames"][0]["transform_matrix"][0][0] = 2
        write_json(path, metadata)
    output = tmp_path / "subset"
    with pytest.raises(ValueError):
        build_subset(source, meshes, output, cohort)
    assert not output.exists()


def test_archive_and_manifest_detect_post_copy_tampering(tmp_path: Path) -> None:
    from zipfile import ZipFile

    source, meshes, cohort, ids = make_source(tmp_path)
    output = tmp_path / "subset"
    build_subset(source, meshes, output, cohort)
    archive = tmp_path / "subset.zip"
    archive_subset(output, archive)
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert "subset/subset_manifest.json" in bundle.namelist()
        assert (
            len([name for name in bundle.namelist() if name.endswith("model.glb")]) == 5
        )
    with pytest.raises(FileExistsError):
        archive_subset(output, archive)
    with pytest.raises(ValueError, match="之外"):
        archive_subset(output, output / "bad.zip")
    image = output / "scenes" / ids[0] / "images/0.png"
    image.write_bytes(image.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        verify_subset(output)

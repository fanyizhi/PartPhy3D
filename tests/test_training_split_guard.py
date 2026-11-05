from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tactile3d.training.dataset import prepare_sft_dataset


def test_explicit_sft_splits_cannot_leak_parts_of_same_object(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8), "white").save(tmp_path / "view.png")
    rows = [
        {
            "sample_id": str(index),
            "group_id": "same_physical_object",
            "images": ["view.png"],
            "split": split,
            "expert_target": {"object_category": "known_test_fixture"},
        }
        for index, split in enumerate(("train", "test"))
    ]
    path = tmp_path / "annotations.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = prepare_sft_dataset(path, tmp_path / "sft")
    assert report["counts"] == {"train": 1, "val": 0, "test": 0}
    assert report["num_rejected"] == 1
    assert "跨分割" in report["rejected"][0]["error"]

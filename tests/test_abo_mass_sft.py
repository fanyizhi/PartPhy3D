"""核对真实质量数据准备的标签键、划分和测试答案隔离。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from PIL import Image


def test_mass_sft_uses_asin_labels_and_hides_test_answer(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts/prepare_abo_mass_sft.py"
    spec = importlib.util.spec_from_file_location("prepare_mass", script)
    assert spec and spec.loader
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "raw"
    root.mkdir()
    splits = {"train": ["A_market"], "val": ["B_market"], "test": ["C_market"]}
    (root / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    (root / "filtered_product_weights.json").write_text(
        json.dumps({"A": 1.0, "B": 2.0, "C": 3.0}), encoding="utf-8"
    )
    for scene, color in zip(
        ("A_market", "B_market", "C_market"), ("red", "green", "blue"), strict=True
    ):
        images = root / "scenes" / scene / "images"
        images.mkdir(parents=True)
        Image.new("RGBA", (32, 32), color).save(images / f"{scene}_00.png")
    output = tmp_path / "prepared"
    report = module.prepare(root, output, "/data/prepared")
    assert report["splits"] == {"train": 1, "val": 1, "test": 1}
    test = json.loads((output / "test_inputs.jsonl").read_text(encoding="utf-8"))
    assert set(test) == {"group_id", "scene_id", "images"}
    train = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    answer = train["messages"][-1]["content"][0]["text"]
    assert json.loads(answer) == {"mass_kg": 1.0}

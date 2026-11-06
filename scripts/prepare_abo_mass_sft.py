"""按ABO官方对象划分准备真实质量SFT，不合成材料或内部结构标签。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.physics.part_io import file_sha256, write_json

SIZE = 384
SYSTEM = "Estimate product mass from visual evidence. Return JSON only."
PROMPT = (
    "Estimate the total mass of this product in kilograms. No measured size or "
    "material information is supplied. Return only a JSON object with one "
    'positive numeric field: "mass_kg".'
)


def prepare(root: Path, output: Path, runtime_root: str) -> dict[str, Any]:
    """隔离物理对象和重复图像，生成白底派生图与可迁移训练记录。

    Args:
        root: 完整ABO-500目录。
        output: 新建的本地输出目录。
        runtime_root: 服务器解压目录，供JSONL中的绝对图像路径使用。

    Returns:
        官方划分数量、输入哈希及明确的监督范围。
    """
    if output.exists():
        raise FileExistsError(f"拒绝覆盖数据：{output}")
    if not runtime_root.startswith("/") or ".." in runtime_root.split("/"):
        raise ValueError("runtime-root必须为Linux绝对路径")
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    weights = json.loads(
        (root / "filtered_product_weights.json").read_text(encoding="utf-8")
    )
    groups = {s: {i.split("_")[0] for i in splits[s]} for s in ("train", "val", "test")}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[first] & groups[second]:
            raise ValueError(f"官方划分有跨集合ASIN重叠：{first}/{second}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    hashes: dict[str, str] = {}
    records, labels, provenance = {}, {}, []
    for split in ("train", "val", "test"):
        rows = []
        for scene in splits[split]:
            # 官方完整权重表使用ASIN，渲染场景名还可能带市场后缀。
            mass = float(weights[scene.split("_")[0]])
            if scene in weights and not math.isclose(
                float(weights[scene]), mass, rel_tol=1e-8
            ):
                raise ValueError(f"场景与ASIN的质量标签冲突：{scene}")
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError(f"质量标签无效：{scene}")
            source = root / "scenes" / scene / "images" / f"{scene}_00.png"
            target = output / "images" / f"{scene}.png"
            with Image.open(source) as image:
                rgba = image.convert("RGBA")
                image = Image.alpha_composite(
                    Image.new("RGBA", rgba.size, "white"), rgba
                ).convert("RGB")
                image.resize((SIZE, SIZE), Image.Resampling.LANCZOS).save(target)
            digest = file_sha256(target)
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f"跨集合重复图像：{scene}")
            hashes[digest] = split
            row: dict[str, Any] = {
                "group_id": scene.split("_")[0],
                "scene_id": scene,
                "images": [f"{runtime_root}/images/{scene}.png"],
            }
            if split != "test":
                row["messages"] = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": PROMPT},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": json.dumps({"mass_kg": mass})}
                        ],
                    },
                ]
            rows.append(row)
            labels[scene] = mass
            provenance.append(
                {
                    "scene_id": scene,
                    "split": split,
                    "source_sha256": file_sha256(source),
                    "derived_sha256": digest,
                }
            )
        records[split] = rows
        name = f"{split}.jsonl" if split != "test" else "test_inputs.jsonl"
        (output / name).write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
    write_json(output / "evaluation_labels.json", labels)
    report = {
        "scope": "ABO_metadata_mass_supervision_only",
        "splits": {s: len(r) for s, r in records.items()},
        "split_sha256": file_sha256(root / "splits.json"),
        "labels_source_sha256": file_sha256(root / "filtered_product_weights.json"),
        "system": SYSTEM,
        "prompt": PROMPT,
        "image_size": SIZE,
        "views_per_object": 1,
        "cross_split_ASIN_overlap": False,
        "cross_split_identical_images": False,
        "physical_parameter_labels_available": False,
        "prior_development_policy": (
            "exclude_first_10_historical_test_ids_in_supplement"
        ),
        "images": provenance,
    }
    write_json(output / "dataset_manifest.json", report)
    archive = output.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with ZipFile(archive, "x", ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
    return {
        "splits": report["splits"],
        "archive": str(archive),
        "archive_sha256": file_sha256(archive),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output, args.runtime_root), indent=2))


if __name__ == "__main__":
    main()

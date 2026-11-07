"""从ABO官方商品归档提取独立长度信息，禁止将质量标签放进推理输入。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile3d.physics.part_io import file_sha256, write_json

UNIT_TO_M = {
    "inches": 0.0254,
    "inch": 0.0254,
    "centimeters": 0.01,
    "centimeter": 0.01,
    "millimeters": 0.001,
    "millimeter": 0.001,
    "meters": 1.0,
    "meter": 1.0,
    "feet": 0.3048,
}
MARKETPLACES = {"ATVPDKIKX0DER": "amazon.com", "A1F83G8C2ARO7P": "amazon.co.uk"}


def metric_dimensions(row: dict[str, Any]) -> dict[str, float] | None:
    dimensions = row.get("item_dimensions", {})
    result = {}
    for axis in ("length", "width", "height"):
        value = dimensions.get(axis, {})
        value = value.get("normalized_value", value)
        if not isinstance(value, dict) or value.get("unit") not in UNIT_TO_M:
            return None
        length = value.get("value")
        if isinstance(length, bool) or not isinstance(length, (int, float)):
            return None
        length = float(length) * UNIT_TO_M[value["unit"]]
        if not np.isfinite(length) or length <= 0:
            return None
        result[axis] = length
    return result


def prepare(archive: Path, cohort: Path, output: Path) -> dict[str, Any]:
    """优先匹配ASIN及市场；只导出尺寸，不以质量筛选或拟合尺度。"""
    ids = json.loads(cohort.read_text(encoding="utf-8"))["object_ids"]
    asins = {scene.split("_")[0] for scene in ids}
    matches: dict[str, list[dict[str, Any]]] = {asin: [] for asin in asins}
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle:
            if not member.isfile() or not member.name.endswith(".json.gz"):
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("归档成员不可读取")
            with gzip.GzipFile(fileobj=stream) as data:
                for line in data:
                    row = json.loads(line)
                    asin = row.get("item_id")
                    if asin not in asins:
                        continue
                    matches[asin].append(
                        {
                            "item_id": asin,
                            "domain_name": row.get("domain_name"),
                            "3dmodel_id": row.get("3dmodel_id"),
                            "source_archive_member": member.name,
                            "source_record_sha256": hashlib.sha256(line).hexdigest(),
                            "item_dimensions": row.get("item_dimensions"),
                            "dimensions_m": metric_dimensions(row),
                        }
                    )
    records = {}
    for scene in ids:
        components = scene.split("_")
        preferred = (
            MARKETPLACES.get(components[1]) if len(components) > 1 else "amazon.com"
        )
        candidates = matches[components[0]]
        exact = [r for r in candidates if r["domain_name"] == preferred]
        usable = [r for r in (exact or candidates) if r["dimensions_m"] is not None]
        different = {
            tuple(
                round(r["dimensions_m"][axis], 8)
                for axis in ("length", "width", "height")
            )
            for r in usable
        }
        status = (
            "available"
            if len(different) == 1
            else ("conflicting_dimensions" if different else "missing_dimensions")
        )
        records[scene] = {
            "status": status,
            "preferred_domain": preferred,
            "candidates": candidates,
            "max_dimension_m": (
                max(usable[0]["dimensions_m"].values())
                if status == "available"
                else None
            ),
            "source": "ABO_catalog_dimensions_not_independent_lab_measurement",
            "RGB_only_input": False,
        }
    result = {
        "scope": "catalog_dimension_assisted_input_no_mass_labels",
        "source_url": "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar",
        "source_archive_sha256": file_sha256(archive),
        "source_credit": "Amazon.com; ABO, Collins et al., CVPR 2022",
        "license": "CC-BY-4.0",
        "mass_labels_included": False,
        "item_weight_passed_to_inference": False,
        "objects": records,
    }
    if output.exists():
        raise FileExistsError("拒绝覆盖商品尺寸证据")
    write_json(output, result)
    return {scene: r["status"] for scene, r in records.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--cohort", type=Path, default=ROOT / "configs/five_object_pilot.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.cohort, args.output), indent=2))


if __name__ == "__main__":
    main()

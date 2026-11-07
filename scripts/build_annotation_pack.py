"""构建无预测叠图的离线人工标注页面，不自动生成部件真值。"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile3d.physics.part_io import file_sha256, write_json

VIEWS = (0, 10, 20)


def build(fields: Path, data_root: Path, output: Path) -> None:
    """将原图和固定部件字典嵌入HTML，PNG与核验JSON由标注者导出。"""
    report = json.loads((fields / "report.json").read_text(encoding="utf-8"))
    cases = []
    for obj in report["objects"]:
        scene = obj["scene_id"]
        token = json.loads(
            (fields / scene / "part_tokens.json").read_text(encoding="utf-8")
        )
        parts = [
            {
                "index": p["part_index"],
                "part_id": p["part_id"],
                "name": p.get("name", p["part_id"]),
            }
            for p in token["parts"]
        ]
        if (
            not parts
            or len(parts) >= 255
            or sorted(p["index"] for p in parts) != list(range(1, len(parts) + 1))
            or len({p["part_id"] for p in parts}) != len(parts)
        ):
            raise ValueError("需要1至254个唯一部件，索引应连续且从1开始")
        for view in VIEWS:
            path = data_root / "scenes" / scene / "images" / f"{scene}_{view:02d}.png"
            with Image.open(path) as image:
                width, height = image.size
            cases.append(
                {
                    "case_id": f"{scene}_{view:02d}",
                    "scene_id": scene,
                    "view": view,
                    "width": width,
                    "height": height,
                    "image_sha256": file_sha256(path),
                    "parts": parts,
                    "image_data": "data:image/png;base64,"
                    + base64.b64encode(path.read_bytes()).decode("ascii"),
                }
            )
    if not cases:
        raise ValueError("没有可标注的对象和原图")
    template = (ROOT / "configs/annotation_canvas.html").read_text(encoding="utf-8")
    payload = json.dumps({"cases": cases}, ensure_ascii=False).replace("<", "\\u003c")
    output.mkdir(parents=True, exist_ok=False)
    (output / "index.html").write_text(
        template.replace("__PROJECT_JSON__", payload), encoding="utf-8"
    )
    write_json(
        output / "annotation_plan.json",
        {
            "cases": [{k: v for k, v in c.items() if k != "image_data"} for c in cases],
            "human_labels_created": False,
            "predictions_shown": False,
            "source_fields_sha256": file_sha256(fields / "report.json"),
        },
    )
    print(f"离线页面：{output / 'index.html'}；尚无人工部件标注")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fields", type=Path, default=ROOT / "examples/observed_surface_demo"
    )
    parser.add_argument("--root", type=Path, default=ROOT / "data/abo500_five")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/human_annotation_pack"
    )
    args = parser.parse_args()
    build(args.fields, args.root, args.output)


if __name__ == "__main__":
    main()

"""直接问可见定位点的材料，区分可见表皮与被猜测的内部支架。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.vlm_interior import _parse_response_content


def observe(locations: Path, root: Path, model_path: Path, output: Path) -> None:
    """材料观测是VLM假设，不宣称点或材料标签为真值。"""
    from matplotlib.font_manager import findfont

    output.mkdir(parents=True, exist_ok=False)
    model = LocalVLM(model_path, max_tokens=500, max_pixels=384 * 384, quantized=True)
    write_json(output / "model.json", model.provenance)
    for directory in sorted(locations.iterdir()):
        if not directory.is_dir():
            continue
        scene = directory.name
        source = json.loads(
            (directory / "localization.json").read_text(encoding="utf-8")
        )
        dest = output / scene
        dest.mkdir()
        records = []
        for view in source["views"]:
            groups = []
            membership = {}
            for part in view["parts"]:
                point = part.get("point")
                if point is None:
                    continue
                index = next(
                    (
                        i
                        for i, p in enumerate(groups)
                        if np.linalg.norm(np.asarray(point) - p) < 15
                    ),
                    None,
                )
                if index is None:
                    index = len(groups)
                    groups.append(point)
                membership[part["part_id"]] = index + 1
            path = (
                root / "scenes" / scene / "images" / f"{scene}_{view['view']:02d}.png"
            )
            with Image.open(path) as source_image:
                rgba = source_image.convert("RGBA")
                image = Image.alpha_composite(
                    Image.new("RGBA", rgba.size, "white"), rgba
                ).convert("RGB")
            original = image.copy()
            montage = Image.new(
                "RGB", (576, 192 * max(1, (len(groups) + 2) // 3)), "white"
            )
            drawing = ImageDraw.Draw(image)
            font = ImageFont.truetype(findfont("DejaVu Sans"), 28)
            for index, point in enumerate(groups, 1):
                x, y = np.asarray(point) * np.array(image.size) / 1000
                drawing.ellipse(
                    (x - 12, y - 12, x + 12, y + 12), outline="red", width=4
                )
                drawing.text(
                    (x + 16, y - 22),
                    str(index),
                    font=font,
                    fill="red",
                    stroke_width=2,
                    stroke_fill="white",
                )
                patch = original.crop((x - 48, y - 48, x + 48, y + 48)).resize(
                    (192, 192)
                )
                detail = ImageDraw.Draw(patch)
                detail.ellipse((86, 86, 106, 106), outline="red", width=2)
                detail.rectangle((0, 0, 45, 35), fill="white")
                detail.text((5, 0), str(index), font=font, fill="black")
                montage.paste(
                    patch, (((index - 1) % 3) * 192, ((index - 1) // 3) * 192)
                )
            image.save(dest / f"{view['view']:02d}_markers.png")
            prompt = (
                f"There are {len(groups)} numbered red rings in this image. "
                "For each numbered ring, identify ONLY the visible material "
                "at its CENTER. Ignore red ink and do not infer hidden structures. "
                "Use unknown if ambiguous. Allowed materials: fabric, leather, "
                "wood, steel, aluminum, plastic, glass, "
                "rubber, foam, ceramic, silicone, paper_cardboard, unknown. "
                "The second image contains enlarged patches "
                "at the same numbered points. "
                'Return a JSON object with key "materials", a list of objects. '
                'Each object must have "marker" (the integer ID) and "material" '
                "(one allowed string). Use double quotes for JSON keys and strings."
            )
            response = model.generate(
                prompt,
                [image, montage],
                "Inspect visible material at marked image points.",
            )
            (dest / f"{view['view']:02d}_raw.txt").write_text(
                response.text, encoding="utf-8"
            )
            payload = _parse_response_content(response.text)["materials"]
            values = {int(p["marker"]): p["material"] for p in payload}
            if set(values) != set(range(1, len(groups) + 1)):
                raise ValueError("点材料回答不完整")
            records.append(
                {
                    "view": view["view"],
                    "image_sha256": file_sha256(path),
                    "parts": [
                        {
                            "part_id": pid,
                            "marker": index,
                            "observed_material": values[index],
                        }
                        for pid, index in membership.items()
                    ],
                }
            )
            write_json(
                dest / "point_evidence.json",
                {
                    "views": records,
                    "source": "VLM_visible_point_hypotheses_not_measured_labels",
                },
            )
            print("POINT_MATERIALS", scene, view["view"], values, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("data/abo500_five"))
    a = p.parse_args()
    observe(a.locations, a.root, a.model, a.output)


if __name__ == "__main__":
    main()

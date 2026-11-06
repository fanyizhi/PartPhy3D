"""从已执行的表面对应生成部件/物性图及同一资产上的换材示意。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.figures.bound_material import raster_face_ids, shade_face_colors
from tactile3d.physics.materials import load_material_priors
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.physics.surface_edit import edit_surface_fields

SIZE = 300
PART_COLORS = np.array(
    [
        [185, 185, 185, 255],
        [90, 118, 168, 255],
        [201, 106, 106, 255],
        [79, 163, 143, 255],
        [133, 115, 181, 255],
        [209, 161, 75, 255],
        [116, 168, 198, 255],
        [140, 109, 91, 255],
    ],
    dtype=np.uint8,
)
MATERIAL_COLORS = {
    "wood": (155, 110, 65, 255),
    "fabric": (159, 164, 156, 255),
    "leather": (130, 77, 55, 255),
    "steel": (77, 114, 156, 255),
    "aluminum": (160, 177, 194, 255),
    "plastic": (119, 150, 133, 255),
}


def scalar_rgba(values: np.ndarray, low: float, high: float) -> np.ndarray:
    import matplotlib as mpl

    norm = mpl.colors.LogNorm(vmin=low, vmax=high)
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("#BBBBBB")
    return np.rint(cmap(norm(np.ma.masked_invalid(values))) * 255).astype(np.uint8)


def render(
    fields: Path, data_root: Path, output: Path, edit_spec: Path | None
) -> dict[str, Any]:
    """编辑与属性字段共享面标签；拒绝把开放表面片直接当成封闭体积。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((fields / "report.json").read_text(encoding="utf-8"))
    specs = json.loads(edit_spec.read_text(encoding="utf-8")) if edit_spec else {}
    priors = {p.name: p for p in load_material_priors("configs/material_priors.yaml")}
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(
        len(report["objects"]),
        6,
        figsize=(15, 2.9 * len(report["objects"])),
        layout="constrained",
        squeeze=False,
    )
    titles = (
        "Observed RGB",
        "SAM2 proposal overlay",
        "Generated Mesh parts",
        "Density prior (kg/m³)",
        "Young’s modulus prior (Pa)",
        "Material edit schematic",
    )
    edit_figure, edit_axes = plt.subplots(
        len(report["objects"]),
        2,
        figsize=(6.4, 2.8 * len(report["objects"])),
        layout="constrained",
        squeeze=False,
    )
    edits = []
    for row, item in enumerate(report["objects"]):
        scene = item["scene_id"]
        directory = fields / scene
        token = json.loads((directory / "part_tokens.json").read_text(encoding="utf-8"))
        mesh_path = Path(token["mesh"])
        if file_sha256(mesh_path) != token["mesh_sha256"]:
            raise ValueError("Mesh哈希错误")
        mesh = trimesh.load_mesh(mesh_path, process=False)
        with np.load(directory / "surface_fields.npz") as stored:
            labels = stored["labels"]
            agreement = stored["agreement"]
        parts = token["parts"]
        density = np.full(len(labels), np.nan)
        youngs = density.copy()
        material_colors = np.tile(PART_COLORS[0], (len(labels), 1))
        table = []
        for part in parts:
            mask = labels == part["part_index"]
            name = part["surface_material"]
            if name in priors:
                density[mask] = priors[name].density_kg_m3.mean
                youngs[mask] = priors[name].youngs_modulus_pa.mean
            material_colors[mask] = MATERIAL_COLORS.get(
                name, tuple(PART_COLORS[part["part_index"] % len(PART_COLORS)])
            )
            table.append(
                {
                    "part_id": part["part_id"],
                    "name": part.get("name", ""),
                    "material_hypothesis": name,
                    "interior_hypothesis": part.get("interior_material", "unknown"),
                    "labeled_faces": int(mask.sum()),
                    "density_prior_kg_m3": (
                        priors[name].density_kg_m3.mean if name in priors else None
                    ),
                    "youngs_prior_pa": (
                        priors[name].youngs_modulus_pa.mean if name in priors else None
                    ),
                }
            )
        selected = specs.get(scene)
        if selected is None:
            candidates = [p for p in parts if np.any(labels == p["part_index"])]
            if not candidates:
                raise ValueError(f"{scene}没有已标注面，不能假装编辑成功")
            target = max(
                candidates,
                key=lambda p: mesh.area_faces[labels == p["part_index"]].sum(),
            )
            selected = {"part_id": target["part_id"], "new_material": "steel"}
        targets = [p for p in parts if p["part_id"] == selected.get("part_id")]
        if len(targets) != 1 or selected.get("new_material") not in priors:
            raise ValueError(f"{scene}编辑需要唯一部件ID和已配置的材料名称")
        target = targets[0]
        new = priors[selected["new_material"]]
        updated = edit_surface_fields(
            labels,
            density,
            youngs,
            material_colors,
            token,
            target["part_id"],
            new,
            MATERIAL_COLORS.get(new.name, (77, 114, 156, 255)),
        )
        changed = updated.changed
        edited_colors = updated.colors_rgba
        edited_density = updated.density_kg_m3
        edited_youngs = updated.youngs_pa
        if not np.array_equal(edited_colors[~changed], material_colors[~changed]):
            raise ValueError("编辑泄漏到其它部件")
        if not np.array_equal(
            edited_density[~changed], density[~changed], equal_nan=True
        ) or not np.array_equal(
            edited_youngs[~changed], youngs[~changed], equal_nan=True
        ):
            raise ValueError("物性更新越界")
        surface = SimpleNamespace(vertices_m=mesh.vertices, faces=mesh.faces)
        face_ids = raster_face_ids(surface, -55, SIZE)
        colors = PART_COLORS[labels % len(PART_COLORS)]
        rendered = [
            shade_face_colors(surface, c, face_ids)
            for c in (
                colors,
                scalar_rgba(density, 50, 8000),
                scalar_rgba(youngs, 1e5, 2.2e11),
                edited_colors,
            )
        ]
        with Image.open(
            data_root / "scenes" / scene / "images" / f"{scene}_00.png"
        ) as image:
            rgba = image.convert("RGBA").resize((SIZE, SIZE))
            original = np.asarray(
                Image.alpha_composite(
                    Image.new("RGBA", rgba.size, "white"), rgba
                ).convert("RGB")
            )
        with np.load(directory / "view_00.npz") as stored:
            mask = stored["labels"]
        overlay = np.asarray(
            Image.fromarray(mask.astype(np.int32), mode="I").resize(
                (SIZE, SIZE), Image.Resampling.NEAREST
            )
        )
        colored = original.copy()
        visible = overlay > 0
        colored[visible] = np.rint(
            0.55 * original[visible]
            + 0.45 * PART_COLORS[overlay[visible] % len(PART_COLORS), :3]
        ).astype(np.uint8)
        for column, image in enumerate([original, colored, *rendered]):
            axis = axes[row, column]
            axis.imshow(image)
            axis.set(xticks=[], yticks=[])
            if row == 0:
                axis.set_title(titles[column], fontsize=9)
            if column == 0:
                axis.set_ylabel(scene.split("_")[0], fontsize=9)
        axes[row, 5].set_xlabel(
            textwrap.fill(target.get("name", target["part_id"]), width=29)
            + f"\n→ {new.name}",
            fontsize=8,
        )
        before_image = shade_face_colors(surface, material_colors, face_ids)
        after_image = shade_face_colors(surface, edited_colors, face_ids)
        selected_pixels = np.zeros(face_ids.shape, dtype=bool)
        visible_pixels = face_ids >= 0
        selected_pixels[visible_pixels] = changed[face_ids[visible_pixels]]
        if not np.array_equal(
            before_image[~selected_pixels], after_image[~selected_pixels]
        ):
            raise ValueError("编辑改变了目标可见区域以外的像素")
        for column, image in enumerate((before_image, after_image)):
            axis = edit_axes[row, column]
            axis.imshow(image)
            axis.set(xticks=[], yticks=[])
            axis.set_title("Before" if column == 0 else "After", fontsize=9)
            axis.set_xlabel(
                textwrap.fill(target.get("name", target["part_id"]), width=32)
                + (f"\n→ {new.name}" if column else ""),
                fontsize=8,
            )
        child = output / scene
        child.mkdir()
        write_json(child / "part_tokens_before.json", token)
        write_json(child / "part_tokens_after.json", updated.tokens)
        np.savez_compressed(
            child / "physical_fields.npz",
            labels=labels,
            agreement=agreement,
            density_prior_kg_m3=density,
            youngs_prior_pa=youngs,
            edited_density_prior_kg_m3=edited_density,
            edited_youngs_prior_pa=edited_youngs,
        )
        for filename, rgba_values in [
            ("parts.ply", colors),
            ("material_before.ply", material_colors),
            ("material_after.ply", edited_colors),
        ]:
            colored_mesh = mesh.copy()
            colored_mesh.visual.face_colors = rgba_values
            colored_mesh.export(child / filename)
        with (child / "part_parameters.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        edit = {
            "scene_id": scene,
            **selected,
            "changed_faces": int(changed.sum()),
            "new_density_prior_kg_m3": new.density_kg_m3.mean,
            "new_youngs_prior_pa": new.youngs_modulus_pa.mean,
            "geometry_unchanged": True,
            "unselected_faces_changed": 0,
            "mass_recomputed": False,
            "mass_blocker": "no_verified_metric_scale_or_closed_material_regions",
            "selection": (
                "user_spec" if scene in specs else "largest_labeled_surface_area_demo"
            ),
        }
        write_json(child / "edit_spec.json", selected)
        write_json(child / "edit_report.json", edit)
        edits.append(edit)
    for column, low, high in ((3, 50, 8000), (4, 1e5, 2.2e11)):
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(
                norm=matplotlib.colors.LogNorm(low, high), cmap="viridis"
            ),
            ax=axes[:, column].tolist(),
            fraction=0.035,
            pad=0.02,
        )
        colorbar.ax.tick_params(labelsize=7)
    figure.suptitle(
        "Visible-surface hypotheses on GENERATED Mesh | gray = unknown\n"
        "Material-conditioned configuration priors; "
        "no measured physics or verified part masks",
        fontsize=11,
    )
    for suffix in ("svg", "png"):
        figure.savefig(
            output / f"surface_properties_and_edits.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    edit_figure.suptitle(
        "Fixed generated Mesh | schematic material colors\n"
        "Only the selected visible surface and its conditional priors change",
        fontsize=10,
    )
    for suffix in ("svg", "png"):
        edit_figure.savefig(
            output / f"part_edit_comparison.{suffix}", dpi=300, bbox_inches="tight"
        )
    plt.close(edit_figure)
    summary = {
        "objects": edits,
        "source_fields_sha256": file_sha256(fields / "report.json"),
        "physics_accuracy_established": False,
        "segmentation_accuracy_established": False,
        "interior_physics_established": False,
        "material_color_is_schematic": True,
    }
    write_json(output / "report.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fields", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("data/abo500_five"))
    p.add_argument("--edit-spec", type=Path)
    args = p.parse_args()
    print(
        json.dumps(
            render(args.fields, args.root, args.output, args.edit_spec), indent=2
        )
    )


if __name__ == "__main__":
    main()

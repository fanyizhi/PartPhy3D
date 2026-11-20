"""Evaluate mesh-projected Ours material labels on overlapping PhysSeg views."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import trimesh
from PIL import Image

from prepare_abo50_benchmark import digest, read_json, write_json
from run_abo50_density_maps import CASES, SIZE, rasterize_faces


def evaluate(args: argparse.Namespace) -> None:
    classes = ("fabric", "wood")
    meshes = read_json(args.density_run / "meshes.json")
    rows = []
    with ZipFile(args.physseg) as archive:
        masks = {}
        for path in archive.namelist():
            fields = path.split("/")
            if (
                len(fields) == 5
                and fields[2] in CASES
                and fields[4] in {f"{name}.png" for name in classes}
            ):
                masks.setdefault((fields[2], int(fields[3])), {})[
                    Path(path).stem
                ] = path
        for scene in CASES:
            entries = [
                (view, paths)
                for (case, view), paths in sorted(masks.items())
                if case == scene
            ]
            if not entries:
                continue
            source = args.observations / "scenes" / scene
            camera = read_json(source / "transforms.json")
            mesh_path = Path(meshes[scene]["path"])
            if digest(mesh_path) != meshes[scene]["sha256"]:
                raise ValueError(f"Mesh changed for {scene}")
            mesh = trimesh.load_mesh(mesh_path, process=False)
            directory = args.density_run / "surface" / "fields" / scene
            tokens = read_json(directory / "part_tokens.json")
            if tokens["mesh_sha256"] != meshes[scene]["sha256"]:
                raise ValueError(f"Surface labels use a different mesh for {scene}")
            with np.load(directory / "surface_fields.npz") as field:
                labels = field["labels"]
            for view, paths in entries:
                frame = camera["frames"][view]
                ids, _ = rasterize_faces(
                    mesh, np.asarray(frame["transform_matrix"]), camera, SIZE
                )
                target = {}
                for name in classes:
                    with Image.open(io.BytesIO(archive.read(paths[name]))) as mask:
                        if (
                            mask.size != (camera["w"], camera["h"])
                            or "A" not in mask.getbands()
                        ):
                            raise ValueError(
                                f"Mask differs from calibrated image for {scene} view {view}"
                            )
                        alpha = mask.getchannel("A").resize(
                            (SIZE, SIZE), Image.Resampling.NEAREST
                        )
                        target[name] = np.asarray(alpha) >= 128
                annotated = target[classes[0]] ^ target[classes[1]]
                visible = annotated & (ids >= 0)
                face_label = np.zeros(ids.shape, dtype=labels.dtype)
                face_label[visible] = labels[ids[visible]]
                counts = {}
                for name in classes:
                    class_parts = [
                        p["part_index"]
                        for p in tokens["parts"]
                        if p["surface_material"] == name
                    ]
                    expected = target[name] & annotated
                    predicted = np.isin(face_label, class_parts) & annotated
                    counts[name] = {
                        "intersection": int(np.count_nonzero(expected & predicted)),
                        "union": int(np.count_nonzero(expected | predicted)),
                    }
                rows.append(
                    {
                        "scene_id": scene,
                        "view": view,
                        "counts": counts,
                        "annotated_pixels": int(annotated.sum()),
                        "predicted_pixels": int(np.count_nonzero(face_label)),
                    }
                )
                # Export the same pixels used for scoring at two fixed camera ranks.
                if args.figure_data and view in (entries[0][0], entries[2][0]):
                    args.figure_data.mkdir(parents=True, exist_ok=True)
                    with Image.open(source / frame["file_path"]) as raw:
                        rgb = raw.convert("RGBA").resize(
                            (SIZE, SIZE), Image.Resampling.LANCZOS
                        )
                    rgb = np.asarray(rgb)
                    target_class = np.zeros(ids.shape, np.uint8)
                    predicted_class = np.zeros_like(target_class)
                    modulus = np.full(ids.shape, np.nan, dtype=np.float32)
                    for index, name in enumerate(classes, 1):
                        target_class[target[name] & annotated] = index
                    # Steel is a prediction outside the two annotated categories.
                    display = ("fabric", "wood", "steel")
                    for part in tokens["parts"]:
                        name = part["surface_material"]
                        pixels = (ids >= 0) & (
                            labels[ids.clip(min=0)] == part["part_index"]
                        )
                        predicted_class[pixels] = display.index(name) + 1
                        modulus[pixels] = tokens["material_priors"][name][
                            "youngs_mean_pa"
                        ]
                    np.savez_compressed(
                        args.figure_data / f"{scene}_{view:02d}.npz",
                        rgba=rgb,
                        target=target_class,
                        prediction=predicted_class,
                        modulus_pa=modulus,
                    )
    if len(rows) != 10 or len({row["scene_id"] for row in rows}) != 2:
        raise ValueError(
            "Expected two frozen overlapping scenes with five labelled views each"
        )
    per_class = {
        name: sum(r["counts"][name]["intersection"] for r in rows)
        / sum(r["counts"][name]["union"] for r in rows)
        for name in classes
    }
    write_json(
        args.output,
        {
            "annotation_archive_sha256": digest(args.physseg),
            "evaluation_area": "pixels belonging to exactly one released class mask",
            "view_count": len(rows),
            "scene_count": 2,
            "class_iou": per_class,
            "miou": float(np.mean(list(per_class.values()))),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "density-run", "physseg", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--figure-data", type=Path)
    evaluate(parser.parse_args())

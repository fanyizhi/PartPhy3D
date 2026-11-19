"""Compare frozen and input-view-calibrated geometry on untouched views."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from prepare_abo50_benchmark import digest, read_json, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.vision.silhouette import iou, silhouette
from tactile3d.vision.silhouette_alignment import fit_silhouette_extents

RESOLUTION = 256


def draw_examples(examples: list[dict], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        1, len(examples), figsize=(10, 3.75), layout="constrained"
    )
    colors = ["#007C83", "#CF3D32", "#328B43"]
    for axis, example in zip(axes, examples):
        axis.imshow(example["image"])
        for mask, color in zip(example["masks"], colors):
            axis.contour(mask, levels=[0.5], colors=[color], linewidths=1.1)
        axis.set_title(
            f"Held-out view {example['index']:02d}\nIoU {example['raw']:.2f} → {example['refined']:.2f}"
        )
        axis.set_axis_off()
    figure.legend(
        handles=[
            Line2D([0], [0], color=color, label=label)
            for color, label in zip(
                colors, ["Observed", "Frozen InstantMesh", "Calibrated mesh"]
            )
        ],
        loc="outside lower center",
        frameon=False,
        ncol=3,
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def evaluate(run: Path, output: Path, figure: bool = False) -> dict:
    protocol = read_json(run / "protocol.json")
    report = read_json(run / "report.json")
    camera_path = Path(protocol["camera_file"])
    mesh_path = run / "camera_space.ply"
    if (
        digest(camera_path) != protocol["camera_sha256"]
        or digest(mesh_path) != report["mesh_sha256"]
    ):
        raise ValueError(
            "Camera or generated mesh differs from reconstruction provenance"
        )
    camera = read_json(camera_path)
    mesh = trimesh.load_mesh(mesh_path, process=False)
    frames = camera["frames"]
    input_ids = protocol["coordinate_transform"]["selected_frame_indices"]
    poses = np.array([frame["transform_matrix"] for frame in frames])
    targets = []
    boxes = {}
    for index, frame in enumerate(frames):
        path = camera_path.parent / frame["file_path"]
        with Image.open(path) as image:
            mask = image.getchannel("A")
            if index in input_ids:
                y, x = np.nonzero(np.asarray(mask) >= 128)
                if not len(y):
                    raise ValueError(f"Empty calibration foreground at view {index}")
                boxes[index] = [x.min(), y.min(), x.max(), y.max()]
            targets.append(np.asarray(mask.resize((RESOLUTION, RESOLUTION))) >= 128)

    refined, fit = fit_silhouette_extents(
        mesh.vertices, poses[input_ids], camera, [boxes[index] for index in input_ids]
    )
    heldout = []
    examples = []
    heldout_ids = [index for index in range(len(frames)) if index not in input_ids]
    example_ids = {
        heldout_ids[i] for i in np.linspace(0, len(heldout_ids) - 1, 3).astype(int)
    }
    for index in heldout_ids:
        target = targets[index]
        raw = silhouette(mesh.vertices, mesh.faces, poses[index], camera, RESOLUTION)
        corrected = silhouette(refined, mesh.faces, poses[index], camera, RESOLUTION)
        heldout.append(
            {
                "index": index,
                "frozen_iou": iou(raw, target),
                "refined_iou": iou(corrected, target),
            }
        )
        if figure and index in example_ids:
            with Image.open(camera_path.parent / frames[index]["file_path"]) as image:
                rgba = image.convert("RGBA").resize((RESOLUTION, RESOLUTION))
                picture = Image.alpha_composite(
                    Image.new("RGBA", rgba.size, "white"), rgba
                )
            examples.append(
                {
                    "index": index,
                    "image": picture.convert("RGB"),
                    "masks": [target, raw, corrected],
                    "raw": heldout[-1]["frozen_iou"],
                    "refined": heldout[-1]["refined_iou"],
                }
            )
    output.mkdir()
    corrected_mesh = mesh.copy()
    corrected_mesh.vertices = refined
    corrected_mesh.export(output / "corrected_camera_space.ply")
    result = {
        "scope": "relative camera-space geometry. No absolute metric scale or mass inferred",
        "input_view_indices": input_ids,
        "fit": fit,
        "heldout_count": len(heldout),
        "mean_frozen_iou": float(np.mean([row["frozen_iou"] for row in heldout])),
        "mean_refined_iou": float(np.mean([row["refined_iou"] for row in heldout])),
        "heldout": heldout,
    }
    write_json(output / "report.json", result)
    if figure:
        draw_examples(examples, output / "heldout_comparison.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path)
    mode.add_argument("--reconstructions", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    if args.reconstructions is not None:
        if args.cohort is None or args.figure or args.workers < 1:
            parser.error(
                "Batch mode requires --cohort and positive workers, without --figure"
            )
        evaluate_batch(args.reconstructions, args.cohort, args.output, args.workers)
        return
    result = evaluate(args.run, args.output, figure=args.figure)
    for name in ("mean_frozen_iou", "mean_refined_iou"):
        print(f"{name}: {result[name]:.6f}")


def evaluate_batch(
    reconstructions: Path, cohort_path: Path, output: Path, workers: int
) -> None:
    output.mkdir()
    ids = read_json(cohort_path)["object_ids"]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError("Expected 50 distinct frozen scene IDs")
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {
            scene: pool.submit(evaluate, reconstructions / scene, output / scene)
            for scene in ids
        }
        for scene, job in futures.items():
            try:
                result = job.result()
                row = {
                    "scene_id": scene,
                    "frozen_iou": result["mean_frozen_iou"],
                    "refined_iou": result["mean_refined_iou"],
                }
            except (OSError, TypeError, KeyError, ValueError, RuntimeError) as error:
                row = {"scene_id": scene, "error": str(error)}
            rows.append(row)
            write_json(output / "progress.json", {"requested": len(ids), "rows": rows})
            print(scene, row, flush=True)
    valid = [row for row in rows if "error" not in row]
    differences = np.array([row["refined_iou"] - row["frozen_iou"] for row in valid])
    aggregate = None
    if len(differences):
        rng = np.random.default_rng(17)
        boot = rng.choice(differences, (2000, len(differences)), replace=True)
        aggregate = {
            "mean_paired_difference": float(differences.mean()),
            "paired_bootstrap_ci95": np.quantile(
                boot.mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "objects_improved": int(np.count_nonzero(differences > 0)),
        }
    write_json(
        output / "summary.json",
        {
            "requested": len(ids),
            "successful": len(valid),
            "aggregate": aggregate,
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()

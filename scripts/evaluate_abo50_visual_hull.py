"""Compare calibrated reconstruction and silhouette carving on the same input views."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from prepare_abo50_benchmark import digest, read_json, write_json
from run_abo50_density_maps import CASES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.vision.silhouette import iou, silhouette
from tactile3d.vision.visual_hull import build_visual_hull

SIZE = 256


def evaluate(
    benchmark: Path,
    output: Path,
    scene: str,
    resolution: int,
    extra_views: int,
    adaptive_threshold: float | None,
) -> dict:
    source = benchmark / "observations" / "scenes" / scene
    run = benchmark / "reconstructions" / scene
    protocol = read_json(run / "protocol.json")
    original_ids = protocol["coordinate_transform"]["selected_frame_indices"]
    camera_path = source / "transforms.json"
    provenance = read_json(benchmark / "observations" / "cohort.json")["files_sha256"]
    if digest(camera_path) != protocol["camera_sha256"]:
        raise ValueError("Camera differs from the six-view reconstruction")
    camera = read_json(camera_path)
    poses = np.array([frame["transform_matrix"] for frame in camera["frames"]])
    ids = list(original_ids)
    centers = poses[:, :3, 3]
    direction = centers - centers.mean(axis=0)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    angular_similarity = direction @ direction.T
    for _ in range(extra_views):
        score = angular_similarity[:, ids].max(axis=1)
        score[ids] = np.inf
        ids.append(int(np.argmin(score)))
    inputs, targets = {}, []
    for index, frame in enumerate(camera["frames"]):
        path = source / frame["file_path"]
        if digest(path) != provenance[f"scenes/{scene}/{frame['file_path']}"]:
            raise ValueError(f"Image {index} changed after cohort freeze")
        with Image.open(path) as image:
            alpha = image.getchannel("A")
            if index in ids:
                inputs[index] = np.asarray(alpha) >= 128
            targets.append(np.asarray(alpha.resize((SIZE, SIZE))) >= 128)
    original = trimesh.load_mesh(
        benchmark / "silhouettes" / scene / "corrected_camera_space.ply", process=False
    )
    input_iou = None
    if adaptive_threshold is not None:
        input_iou = float(
            np.mean(
                [
                    iou(
                        silhouette(
                            original.vertices, original.faces, poses[i], camera, SIZE
                        ),
                        targets[i],
                    )
                    for i in original_ids
                ]
            )
        )
    selected = input_iou is None or input_iou < adaptive_threshold
    candidate, pitch = (
        build_visual_hull(poses[ids], camera, [inputs[i] for i in ids], resolution)
        if selected
        else (original, None)
    )
    method_name = "visual_hull" if adaptive_threshold is None else "adaptive"
    rows = []
    for index, target in enumerate(targets):
        if index in ids:
            continue
        row = {"view": index}
        for method, mesh in (("instantmesh", original), (method_name, candidate)):
            if method == method_name and not selected:
                row[method] = row["instantmesh"]
                continue
            predicted = silhouette(
                mesh.vertices, mesh.faces, poses[index], camera, SIZE
            )
            overlap = np.count_nonzero(predicted & target)
            row[method] = {
                "iou": iou(predicted, target),
                "recall": overlap / np.count_nonzero(target),
                "precision": overlap / max(np.count_nonzero(predicted), 1),
            }
        rows.append(row)
    destination = output / scene
    destination.mkdir()
    hull_sha = None
    if selected:
        hull_path = destination / "visual_hull.ply"
        candidate.export(hull_path)
        hull_sha = digest(hull_path)
    result = {
        "scene_id": scene,
        "input_views": ids,
        "instantmesh_input_views": original_ids,
        "instantmesh_input_iou": input_iou,
        "visual_hull_selected": selected,
        "pitch": pitch,
        "method_face_counts": {
            "instantmesh": len(original.faces),
            method_name: len(candidate.faces),
        },
        "visual_hull_sha256": hull_sha,
        "rows": rows,
    }
    for name in ("instantmesh", method_name):
        result[name] = {
            metric: float(np.mean([r[name][metric] for r in rows]))
            for metric in ("iou", "recall", "precision")
        }
    write_json(destination / "report.json", result)
    return {key: value for key, value in result.items() if key != "rows"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--resolution", type=int, default=144)
    parser.add_argument("--extra-views", type=int, choices=range(19), default=0)
    parser.add_argument("--adaptive-threshold", type=float)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.adaptive_threshold is not None and not 0 < args.adaptive_threshold < 1:
        parser.error("--adaptive-threshold must be between zero and one")
    cohort = read_json(args.benchmark / "observations" / "cohort.json")["object_ids"]
    cases = cohort if args.all_scenes else CASES
    if not set(cases) <= set(cohort):
        raise ValueError("Selected objects are outside the frozen cohort")
    args.output.mkdir()
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {
            s: pool.submit(
                evaluate,
                args.benchmark,
                args.output,
                s,
                args.resolution,
                args.extra_views,
                args.adaptive_threshold,
            )
            for s in cases
        }
        for scene, job in jobs.items():
            try:
                rows.append(job.result())
            except (OSError, KeyError, ValueError, RuntimeError) as error:
                rows.append({"scene_id": scene, "error": str(error)})
            print(rows[-1], flush=True)
            write_json(
                args.output / "progress.json", {"requested": len(cases), "scenes": rows}
            )
    scored = [row for row in rows if "error" not in row]
    metrics = {}
    method_name = "visual_hull" if args.adaptive_threshold is None else "adaptive"
    for name in ("instantmesh", method_name):
        if scored:
            metrics[name] = {
                k: float(np.mean([r[name][k] for r in scored])) for k in scored[0][name]
            }
    write_json(
        args.output / "summary.json",
        {
            "requested": len(cases),
            "valid": len(scored),
            "metrics": metrics,
            "adaptive_threshold": args.adaptive_threshold,
            "scenes": rows,
        },
    )


if __name__ == "__main__":
    main()

"""在同一真实图像特征上比较冻结前端的全量与表面单元稀疏权重解码。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_observed_instantmesh import IMAGE_SIZE, load_frozen_reconstructor

from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.research.observed_cameras import calibrated_observed_cameras
from tactile3d.research.sparse_decoder import decode_surface_weights

REPETITIONS = 3
VERTEX_ATOL = 1e-5


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """同特征、同分辨率、交替重复比较；拓扑或顶点超差就报告失败。"""
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    model = load_frozen_reconstructor(
        args.repository, args.checkpoint, ROOT / "configs/pretrained/dino-vitb16"
    )
    ids = json.loads((ROOT / "configs/five_object_pilot.json").read_text())[
        "object_ids"
    ]
    results = []
    decoder = model.synthesizer.decoder
    original = decoder.get_geometry_prediction
    stats: list[dict[str, Any]] = []

    def sparse(self: Any, features: Any, indices: Any) -> tuple[Any, Any, Any]:
        return decode_surface_weights(self, features, indices, stats)

    for scene in ids[: args.limit]:
        directory = args.root / "scenes" / scene
        camera = json.loads((directory / "transforms.json").read_text())
        indices, cameras, _ = calibrated_observed_cameras(camera)
        images, hashes = [], {}
        for index in indices:
            path = (directory / camera["frames"][index]["file_path"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("图像路径越出场景目录")
            hashes[path.name] = file_sha256(path)
            with Image.open(path) as source:
                rgba = source.convert("RGBA")
                rgb = (
                    Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
                    .convert("RGB")
                    .resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
                )
                images.append(np.asarray(rgb, dtype=np.float32) / 255)
        tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
        with torch.inference_mode():
            planes = model.forward_planes(
                tensor[None].to("cuda"), torch.from_numpy(cameras)[None].to("cuda")
            )
            timings: dict[str, list[float]] = {"dense": [], "sparse": []}
            peaks: dict[str, list[int]] = {"dense": [], "sparse": []}
            references = {}
            for repeat in range(REPETITIONS + 1):
                for mode in ("dense", "sparse"):
                    decoder.get_geometry_prediction = (
                        original if mode == "dense" else MethodType(sparse, decoder)
                    )
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    try:
                        value = model.get_geometry_prediction(planes)
                    finally:
                        decoder.get_geometry_prediction = original
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if repeat:
                        timings[mode].append(elapsed)
                        peaks[mode].append(torch.cuda.max_memory_allocated())
                    if repeat == 0:
                        references[mode] = (
                            value[0][0].cpu().numpy(),
                            value[1][0].cpu().numpy(),
                        )
                    del value
            before_v, before_f = references["dense"]
            after_v, after_f = references["sparse"]
            same_faces = np.array_equal(before_f, after_f)
            same_shape = before_v.shape == after_v.shape
            vertex_error = (
                float(np.max(np.abs(before_v - after_v))) if same_shape else None
            )
            result = {
                "scene_id": scene,
                "input_hashes": hashes,
                "dense_faces": len(before_f),
                "sparse_faces": len(after_f),
                "topology_identical": same_faces,
                "max_vertex_error": vertex_error,
                "geometry_equivalent": bool(
                    same_faces and same_shape and vertex_error <= VERTEX_ATOL
                ),
                "seconds": timings,
                "peak_allocated_bytes": peaks,
                "median_decoder_speedup": float(
                    np.median(timings["dense"]) / np.median(timings["sparse"])
                ),
                "sparse_counts": stats[-1],
                "full_frontend_speedup_measured": False,
            }
            child = args.output / scene
            child.mkdir()
            np.savez_compressed(
                child / "geometry_equivalence.npz",
                dense_vertices=before_v,
                dense_faces=before_f,
                sparse_vertices=after_v,
                sparse_faces=after_f,
            )
            write_json(child / "report.json", result)
            results.append(result)
            write_json(
                args.output / "report.json",
                {
                    "scope": "generation_stage_surface_cell_weight_pruning",
                    "upstream": "InstantMesh_08822c52fdc399b93ea00e4fa9e596344ed52ccc",
                    "checkpoint_sha256": file_sha256(args.checkpoint),
                    "repetitions_after_warmup": REPETITIONS,
                    "physical_interface_budget_implemented": False,
                    "sdf_vertex_queries_reduced": False,
                    "meshes_simplified": False,
                    "objects": results,
                },
            )
            print(
                "SPARSE_DECODER",
                scene,
                result["geometry_equivalent"],
                result["median_decoder_speedup"],
                flush=True,
            )
            del planes
    return {"objects": results}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--root", type=Path, default=ROOT / "data/abo500_five")
    p.add_argument("--limit", type=int, default=1, choices=range(1, 6))
    print(json.dumps(benchmark(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()

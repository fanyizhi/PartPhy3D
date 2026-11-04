from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tactile3d.utils.io import read_json

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MESH_EXTENSIONS = (".ply", ".obj", ".glb", ".stl", ".off")


@dataclass(frozen=True)
class CameraView:
    image_path: Path
    c2w: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class ABOScene:
    scene_id: str
    scene_dir: Path
    split: str
    mass_kg: float | None
    mesh_path: Path
    mesh_unit_to_meter: float
    views: tuple[CameraView, ...]


class ABO500Dataset:
    """Reader for the ABO-500 release used by NeRF2Physics and PUGS.

    The reader intentionally accepts a few harmless schema variants because the
    released metadata and reconstructed derivatives use both scene-index and ASIN
    keys. A scene is never accepted without a positive mass, camera views, and a
    mesh whose metric scale is known.
    """

    def __init__(
        self,
        root: str | Path,
        mesh_root: str | Path | None = None,
        mesh_unit_to_meter: float | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.scenes_root = self.root / "scenes"
        self.mesh_root = Path(mesh_root).expanduser().resolve() if mesh_root else None
        self.default_mesh_unit_to_meter = mesh_unit_to_meter
        self.weights = self._load_weights()
        self.splits = self._load_splits()

    def scene_ids(self, split: str = "all") -> list[str]:
        if split == "all":
            return sorted(self.weights)
        if split not in self.splits:
            raise KeyError(f"Unknown split {split!r}; available: {sorted(self.splits)}")
        return [scene_id for scene_id in self.splits[split] if scene_id in self.weights]

    def get_scene(self, scene_id: str, split: str | None = None) -> ABOScene:
        scene_dir = self.scenes_root / scene_id
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"ABO-500 scene directory is missing: {scene_dir}")
        if scene_id not in self.weights:
            raise KeyError(f"No ground-truth mass for scene {scene_id}")

        mesh_path = self._resolve_mesh(scene_id, scene_dir)
        unit_to_meter = self._resolve_mesh_scale(scene_dir)
        views = tuple(self._load_views(scene_dir))
        if not views:
            raise ValueError(f"No valid camera views found for scene {scene_id}")
        return ABOScene(
            scene_id=scene_id,
            scene_dir=scene_dir,
            split=split or self._split_for(scene_id),
            mass_kg=float(self.weights[scene_id]),
            mesh_path=mesh_path,
            mesh_unit_to_meter=unit_to_meter,
            views=views,
        )

    def validate(self, split: str = "all", limit: int | None = None) -> dict[str, Any]:
        ids = self.scene_ids(split)
        if limit is not None:
            ids = ids[:limit]
        failures: list[dict[str, str]] = []
        valid = 0
        for scene_id in ids:
            try:
                self.get_scene(scene_id, None if split == "all" else split)
                valid += 1
            except (FileNotFoundError, KeyError, ValueError) as exc:
                failures.append({"scene_id": scene_id, "error": str(exc)})
        return {
            "root": str(self.root),
            "split": split,
            "num_requested": len(ids),
            "num_valid": valid,
            "num_failed": len(failures),
            "failures": failures,
        }

    def _load_weights(self) -> dict[str, float]:
        path = self.root / "filtered_product_weights.json"
        if not path.exists():
            raise FileNotFoundError(f"ABO-500 weight metadata is missing: {path}")
        raw = read_json(path)
        weights: dict[str, float] = {}
        if isinstance(raw, dict):
            iterator: Iterable[tuple[str, Any]] = raw.items()
        elif isinstance(raw, list):
            iterator = ((str(row.get("scene_id", row.get("id", ""))), row) for row in raw)
        else:
            raise TypeError(f"Unsupported weight metadata schema in {path}")
        for scene_id, value in iterator:
            if not scene_id:
                continue
            mass = _parse_mass_kg(value)
            if mass is not None and mass > 0:
                weights[str(scene_id)] = mass
        if not weights:
            raise ValueError(f"No positive masses found in {path}")
        return weights

    def _load_splits(self) -> dict[str, list[str]]:
        path = self.root / "splits.json"
        if not path.exists():
            ids = sorted(self.weights)
            return {
                "train": ids[: int(0.6 * len(ids))],
                "val": ids[int(0.6 * len(ids)) : int(0.8 * len(ids))],
                "test": ids[int(0.8 * len(ids)) :],
            }
        raw = read_json(path)
        splits: dict[str, list[str]] = {}
        if isinstance(raw, dict) and all(isinstance(v, list) for v in raw.values()):
            splits = {str(k): [str(x) for x in v] for k, v in raw.items()}
        elif isinstance(raw, dict):
            for scene_id, split in raw.items():
                splits.setdefault(str(split), []).append(str(scene_id))
        if not splits:
            raise ValueError(f"Unsupported split schema in {path}")
        return splits

    def _resolve_mesh(self, scene_id: str, scene_dir: Path) -> Path:
        candidates: list[Path] = []
        if self.mesh_root is not None:
            candidates.extend(self.mesh_root / f"{scene_id}{ext}" for ext in MESH_EXTENSIONS)
            for name in ("mesh", "object", "reconstruction"):
                candidates.extend(self.mesh_root / scene_id / f"{name}{ext}" for ext in MESH_EXTENSIONS)
        for name in ("mesh", "object", "reconstruction"):
            candidates.extend(scene_dir / f"{name}{ext}" for ext in MESH_EXTENSIONS)
            candidates.extend(scene_dir / "mesh" / f"{name}{ext}" for ext in MESH_EXTENSIONS)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        location = self.mesh_root if self.mesh_root is not None else scene_dir
        raise FileNotFoundError(f"No mesh found for {scene_id} under {location}")

    def _resolve_mesh_scale(self, scene_dir: Path) -> float:
        scale_path = scene_dir / "mesh_scale.json"
        if scale_path.exists():
            raw = read_json(scale_path)
            value = raw.get("mesh_unit_to_meter", raw.get("unit_to_meter"))
            if value is not None and float(value) > 0:
                return float(value)
        if self.default_mesh_unit_to_meter is not None and self.default_mesh_unit_to_meter > 0:
            return float(self.default_mesh_unit_to_meter)
        raise ValueError(
            f"Metric mesh scale is unknown for {scene_dir.name}. Add mesh_scale.json "
            "or pass --mesh-unit-to-meter; normalized mesh volume cannot determine mass."
        )

    def _load_views(self, scene_dir: Path) -> list[CameraView]:
        return load_camera_views(scene_dir)

    def _split_for(self, scene_id: str) -> str:
        for split, ids in self.splits.items():
            if scene_id in ids:
                return split
        return "unknown"


def load_camera_views(scene_dir: str | Path) -> list[CameraView]:
    """Load NeRF-style camera metadata for a rendered mesh scene."""
    scene_dir = Path(scene_dir)
    transform_path = scene_dir / "transforms.json"
    if not transform_path.exists():
        raise FileNotFoundError(f"Camera metadata is missing: {transform_path}")
    meta = read_json(transform_path)
    frames = meta.get("frames", [])
    views: list[CameraView] = []
    for frame in frames:
        image_path = _resolve_image_path(scene_dir, frame.get("file_path", ""))
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
        width = int(frame.get("w", meta.get("w", actual_width)))
        height = int(frame.get("h", meta.get("h", actual_height)))
        fx = _focal(frame, meta, "fl_x", "camera_angle_x", width)
        fy = _focal(frame, meta, "fl_y", "camera_angle_y", height, fallback=fx)
        cx = float(frame.get("cx", meta.get("cx", width / 2.0)))
        cy = float(frame.get("cy", meta.get("cy", height / 2.0)))
        c2w = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if c2w.shape != (4, 4):
            continue
        views.append(
            CameraView(
                image_path=image_path,
                c2w=c2w,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
            )
        )
    return views

def _parse_mass_kg(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("mass_kg", "weight_kg", "mass", "weight", "value"):
            if key in value:
                parsed = _parse_mass_kg(value[key])
                if parsed is None:
                    continue
                unit = str(value.get("unit", value.get("units", "kg"))).lower()
                return _convert_mass(parsed, unit)
        return None
    if isinstance(value, str):
        match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z]*)", value)
        if match:
            return _convert_mass(float(match.group(1)), match.group(2) or "kg")
    return None


def _convert_mass(value: float, unit: str) -> float:
    unit = unit.strip().lower()
    factors = {
        "kg": 1.0,
        "kgs": 1.0,
        "kilogram": 1.0,
        "kilograms": 1.0,
        "g": 1e-3,
        "gram": 1e-3,
        "grams": 1e-3,
        "lb": 0.45359237,
        "lbs": 0.45359237,
        "pound": 0.45359237,
        "pounds": 0.45359237,
        "oz": 0.028349523125,
        "ounce": 0.028349523125,
        "ounces": 0.028349523125,
    }
    return value * factors.get(unit, 1.0)


def _resolve_image_path(scene_dir: Path, file_path: str) -> Path | None:
    raw = Path(file_path.lstrip("./"))
    candidates = [scene_dir / raw, scene_dir / "images" / raw.name]
    if raw.suffix:
        candidates.extend(scene_dir / "images" / f"{raw.stem}{ext}" for ext in IMAGE_EXTENSIONS)
    else:
        candidates.extend((scene_dir / raw).with_suffix(ext) for ext in IMAGE_EXTENSIONS)
        candidates.extend((scene_dir / "images" / raw.name).with_suffix(ext) for ext in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _focal(
    frame: dict[str, Any],
    meta: dict[str, Any],
    focal_key: str,
    angle_key: str,
    size: int,
    fallback: float | None = None,
) -> float:
    if focal_key in frame or focal_key in meta:
        return float(frame.get(focal_key, meta[focal_key]))
    if angle_key in frame or angle_key in meta:
        angle = float(frame.get(angle_key, meta[angle_key]))
        return 0.5 * size / np.tan(0.5 * angle)
    if fallback is not None:
        return float(fallback)
    raise ValueError(f"Missing {focal_key} or {angle_key} in transforms.json")

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


@dataclass
class MeshHypothesis:
    mesh: trimesh.Trimesh
    log_prob: float
    metadata: dict


class MeshReconstructor(ABC):
    """Interface for image-conditioned mesh predictors."""

    @abstractmethod
    def sample(self, image_path: str | Path, rng: np.random.Generator) -> MeshHypothesis:
        """Sample one mesh hypothesis."""


class GeometricImageReconstructor(MeshReconstructor):
    """A watertight geometric baseline derived from image foreground statistics."""

    def __init__(
        self,
        resolution: int = 48,
        nominal_depth_m: float = 0.08,
        scale_jitter: float = 0.08,
        depth_jitter: float = 0.12,
    ) -> None:
        self.resolution = resolution
        self.nominal_depth_m = nominal_depth_m
        self.scale_jitter = scale_jitter
        self.depth_jitter = depth_jitter

    def sample(self, image_path: str | Path, rng: np.random.Generator) -> MeshHypothesis:
        image = Image.open(image_path).convert("RGBA")
        alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        gray = rgb.mean(axis=-1)

        mask = alpha > 0.05
        if mask.mean() > 0.98:
            mask = gray < np.quantile(gray, 0.92)
        if mask.sum() < 16:
            mask = np.ones_like(gray, dtype=bool)

        ys, xs = np.where(mask)
        h, w = gray.shape
        width_ratio = max((xs.max() - xs.min() + 1) / max(w, 1), 0.1)
        height_ratio = max((ys.max() - ys.min() + 1) / max(h, 1), 0.1)
        aspect = width_ratio / max(height_ratio, 1e-6)

        area_ratio = float(mask.mean())
        brightness = float(gray[mask].mean())
        roughness = float(gray[mask].std())

        base_radius = 0.08 * np.sqrt(area_ratio + 0.2)
        sx = base_radius * np.sqrt(aspect)
        sy = base_radius / np.sqrt(aspect)
        sz = self.nominal_depth_m * (0.45 + 0.9 * (1.0 - brightness + roughness))

        scales = np.array([sx, sy, sz], dtype=np.float64)
        scales *= rng.normal(1.0, self.scale_jitter, size=3)
        scales[2] *= rng.normal(1.0, self.depth_jitter)
        scales = np.clip(scales, 0.01, 0.35)

        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(scales)

        elongation = float(abs(np.log(aspect)))
        log_prob = -0.5 * (elongation + roughness)
        return MeshHypothesis(
            mesh=mesh,
            log_prob=log_prob,
            metadata={
                "backend": "geometric",
                "foreground_area_ratio": area_ratio,
                "brightness": brightness,
                "roughness": roughness,
                "scale_xyz_m": scales.tolist(),
            },
        )


def build_reconstructor(cfg: dict) -> MeshReconstructor:
    mesh_cfg = cfg.get("mesh", {})
    backend = mesh_cfg.get("backend", "geometric")
    if backend == "geometric":
        return GeometricImageReconstructor(
            resolution=int(mesh_cfg.get("resolution", 48)),
            nominal_depth_m=float(mesh_cfg.get("nominal_depth_m", 0.08)),
            scale_jitter=float(mesh_cfg.get("scale_jitter", 0.08)),
            depth_jitter=float(mesh_cfg.get("depth_jitter", 0.12)),
        )
    raise ValueError(f"Unknown mesh backend: {backend}")

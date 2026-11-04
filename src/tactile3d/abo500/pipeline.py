from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from tqdm import tqdm

from tactile3d.abo500.dataset import ABO500Dataset, ABOScene
from tactile3d.config import resolve_config_path
from tactile3d.mesh.io import load_triangle_mesh
from tactile3d.mesh.repair import repair_mesh
from tactile3d.mesh.segmentation import PartSegmentation, segment_mesh
from tactile3d.models.calibration import (
    apply_calibration_to_density,
    calibrated_probabilities,
    load_calibration,
    residual_log_sigma,
)
from tactile3d.physics.interior import (
    PartInteriorProposal,
    expected_material_volumes,
    sample_part_physics,
    solid_proposal,
)
from tactile3d.physics.materials import load_material_priors
from tactile3d.utils.io import ensure_dir, write_json, write_jsonl
from tactile3d.vision.material_encoder import MaterialEncoder, MaterialEvidence
from tactile3d.vision.multiview import collect_part_observations
from tactile3d.vision.vlm_interior import InteriorVLMEstimator


class ABO500MeshPipeline:
    def __init__(self, cfg: dict[str, Any], calibration_path: str | Path | None = None) -> None:
        self.cfg = cfg
        prior_path = resolve_config_path(
            cfg.get("materials", {}).get("priors", "configs/material_priors.yaml"),
            cfg.get("_config_dir"),
        )
        self.priors = load_material_priors(prior_path)
        self.material_names = [prior.name for prior in self.priors]
        self.encoder = MaterialEncoder(self.material_names, cfg.get("features", {}))
        self.interior_estimator = InteriorVLMEstimator(
            self.material_names, cfg.get("vlm", {"backend": "surface_only"})
        )
        self.calibration = load_calibration(calibration_path)
        self.rng = np.random.default_rng(int(cfg.get("seed", 17)))

    def infer_scene(self, scene: ABOScene, output_dir: str | Path) -> tuple[dict, dict]:
        output_dir = ensure_dir(output_dir)
        mesh = _load_mesh(scene.mesh_path)
        mesh, repair_report = repair_mesh(
            mesh,
            strategy=str(self.cfg.get("mesh", {}).get("repair_strategy", "fill_only")),
        )
        require_watertight = bool(self.cfg.get("mesh", {}).get("require_watertight", True))
        if require_watertight and not mesh.is_watertight:
            raise ValueError(f"Mesh for {scene.scene_id} is not watertight after repair")
        if not mesh.is_volume or abs(float(mesh.volume)) <= 0:
            raise ValueError(f"Mesh for {scene.scene_id} has no valid enclosed volume")

        segmentation = segment_mesh(
            mesh,
            num_parts=int(self.cfg.get("segmentation", {}).get("num_parts", 6)),
            cfg=self.cfg,
            rng=np.random.default_rng(self.rng.integers(0, 2**31 - 1)),
        )
        view_cfg = self.cfg.get("views", {})
        observations = collect_part_observations(
            mesh,
            segmentation.face_labels,
            scene.views,
            max_views=int(view_cfg.get("max_views", 12)),
            crop_padding=float(view_cfg.get("crop_padding", 0.12)),
            zbuffer_size=int(view_cfg.get("zbuffer_size", 256)),
        )
        evidence = {
            part_id: self.encoder.encode(part_observations)
            for part_id, part_observations in observations.items()
        }

        volume_m3 = abs(float(mesh.volume)) * scene.mesh_unit_to_meter**3
        repair_report["volume_mesh_units3"] = float(abs(mesh.volume))
        repair_report["volume_m3"] = volume_m3
        part_volumes = {
            part_id: volume_m3 * fraction
            for part_id, fraction in segmentation.part_volume_fraction.items()
        }
        area_m2 = float(mesh.area) * scene.mesh_unit_to_meter**2
        part_areas = {
            part_id: area_m2 * fraction
            for part_id, fraction in segmentation.part_area_fraction.items()
        }
        proposals = self._propose_interiors(
            scene,
            part_volumes,
            part_areas,
            observations,
            evidence,
        )
        samples = self._sample_posterior(part_volumes, part_areas, evidence, proposals)
        prediction = self._summarize(
            scene,
            mesh,
            repair_report,
            part_volumes,
            part_areas,
            evidence,
            proposals,
            samples,
        )
        feature_row = self._feature_row(
            scene, part_volumes, part_areas, evidence, proposals
        )
        self._save_artifacts(output_dir, mesh, scene, segmentation, observations, prediction)
        return prediction, feature_row

    def _propose_interiors(
        self,
        scene: ABOScene,
        part_volumes: dict[int, float],
        part_areas: dict[int, float],
        observations: dict[int, list],
        evidence: dict[int, MaterialEvidence],
    ) -> dict[int, PartInteriorProposal]:
        proposals: dict[int, PartInteriorProposal] = {}
        fallback_on_error = bool(self.cfg.get("vlm", {}).get("fallback_on_error", False))
        for part_id, part_volume in part_volumes.items():
            geometry = {
                "enclosed_volume_m3": float(part_volume),
                "surface_area_m2": float(part_areas[part_id]),
            }
            try:
                proposals[part_id] = self.interior_estimator.propose(
                    scene.scene_id,
                    part_id,
                    observations[part_id],
                    evidence[part_id],
                    geometry,
                )
            except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
                if not fallback_on_error:
                    raise
                probabilities = _softmax(evidence[part_id].mean_logits)
                proposals[part_id] = solid_proposal(
                    part_id,
                    self.material_names,
                    probabilities,
                    source=f"fallback_after_{type(exc).__name__}",
                )
        return proposals

    def _sample_posterior(
        self,
        part_volumes: dict[int, float],
        part_areas: dict[int, float],
        evidence: dict[int, MaterialEvidence],
        proposals: dict[int, PartInteriorProposal],
    ) -> dict[str, Any]:
        uncertainty_cfg = self.cfg.get("uncertainty", {})
        num_samples = int(uncertainty_cfg.get("num_samples", 256))
        bootstrap_fraction = float(uncertainty_cfg.get("view_bootstrap_fraction", 0.75))
        scale_cv = max(float(uncertainty_cfg.get("metric_scale_cv", 0.03)), 0.0)
        scale_log_sigma = math.sqrt(math.log1p(scale_cv**2))
        residual_sigma = residual_log_sigma(self.calibration)
        default_temperature = float(self.cfg.get("materials", {}).get("temperature", 1.0))

        part_samples: dict[int, dict[str, Any]] = {}
        for part_id in part_volumes:
            part_samples[part_id] = {
                "density": np.zeros(num_samples),
                "youngs": np.zeros(num_samples),
                "mass": np.zeros(num_samples),
                "occupied_volume": np.zeros(num_samples),
                "material_counts": np.zeros(len(self.priors)),
                "probability_sum": np.zeros(len(self.priors)),
                "material_mass": np.zeros(len(self.priors)),
                "structure_counts": {
                    hypothesis.structure: 0 for hypothesis in proposals[part_id].hypotheses
                },
            }
        object_mass = np.zeros(num_samples)

        for sample_id in range(num_samples):
            metric_scale = _unit_lognormal(self.rng, scale_log_sigma)
            residual_scale = _unit_lognormal(self.rng, residual_sigma)
            for part_id, base_volume in part_volumes.items():
                part_evidence = evidence[part_id]
                num_views = len(part_evidence.view_logits)
                count = max(1, math.ceil(num_views * bootstrap_fraction))
                view_prob = part_evidence.view_weights / part_evidence.view_weights.sum()
                selected = self.rng.choice(num_views, size=count, replace=True, p=view_prob)
                logits = part_evidence.view_logits[selected].mean(axis=0)
                probabilities = calibrated_probabilities(
                    logits,
                    self.material_names,
                    self.calibration,
                    default_temperature,
                )
                probabilities = self._fuse_surface_probabilities(
                    probabilities, proposals[part_id]
                )
                material_id = int(self.rng.choice(len(self.priors), p=probabilities))
                density_scale = apply_calibration_to_density(1.0, self.calibration)
                physics_sample = sample_part_physics(
                    base_volume * metric_scale**3,
                    part_areas[part_id] * metric_scale**2,
                    probabilities,
                    proposals[part_id],
                    self.priors,
                    self.rng,
                    density_scale=density_scale * residual_scale,
                )
                values = part_samples[part_id]
                values["density"][sample_id] = physics_sample.effective_density_kg_m3
                values["youngs"][sample_id] = physics_sample.volume_weighted_youngs_pa
                values["mass"][sample_id] = physics_sample.mass_kg
                values["occupied_volume"][sample_id] = physics_sample.occupied_volume_m3
                values["material_counts"][material_id] += 1
                values["probability_sum"] += probabilities
                values["structure_counts"][physics_sample.structure] += 1
                for material, mass in physics_sample.material_mass_kg.items():
                    values["material_mass"][self.material_names.index(material)] += mass
                object_mass[sample_id] += physics_sample.mass_kg

        return {"object_mass": object_mass, "parts": part_samples, "num_samples": num_samples}

    def _fuse_surface_probabilities(
        self,
        feature_probabilities: np.ndarray,
        proposal: PartInteriorProposal,
    ) -> np.ndarray:
        vlm = np.asarray(
            [proposal.surface_material_probabilities[name] for name in self.material_names],
            dtype=np.float64,
        )
        weight = float(np.clip(self.cfg.get("vlm", {}).get("surface_fusion_weight", 0.5), 0, 1))
        if proposal.source == "surface_only":
            return feature_probabilities
        log_probability = (1.0 - weight) * np.log(np.maximum(feature_probabilities, 1e-12))
        log_probability += weight * np.log(np.maximum(vlm, 1e-12))
        return _softmax(log_probability)

    def _summarize(
        self,
        scene: ABOScene,
        mesh: trimesh.Trimesh,
        repair_report: dict,
        part_volumes: dict[int, float],
        part_areas: dict[int, float],
        evidence: dict[int, MaterialEvidence],
        proposals: dict[int, PartInteriorProposal],
        samples: dict[str, Any],
    ) -> dict[str, Any]:
        parts = []
        for part_id in sorted(part_volumes):
            values = samples["parts"][part_id]
            probability = values["probability_sum"] / samples["num_samples"]
            structure_probability = {
                name: float(count / samples["num_samples"])
                for name, count in sorted(values["structure_counts"].items())
            }
            material_mass = values["material_mass"]
            material_mass_fraction = material_mass / max(float(material_mass.sum()), 1e-12)
            parts.append(
                {
                    "part_id": part_id,
                    "volume_m3": float(part_volumes[part_id]),
                    "enclosed_volume_m3": float(part_volumes[part_id]),
                    "surface_area_m2": float(part_areas[part_id]),
                    "occupied_volume_m3": _stats(values["occupied_volume"]),
                    "num_observed_views": len(evidence[part_id].view_logits),
                    "material_probabilities": {
                        name: float(probability[index])
                        for index, name in enumerate(self.material_names)
                    },
                    "density_kg_m3": _stats(values["density"]),
                    "youngs_modulus_pa": _stats(values["youngs"]),
                    "mass_kg": _stats(values["mass"]),
                    "interior": {
                        "proposal": proposals[part_id].to_json(),
                        "posterior_structure_probabilities": structure_probability,
                        "posterior_material_mass_fractions": {
                            name: float(material_mass_fraction[index])
                            for index, name in enumerate(self.material_names)
                            if material_mass_fraction[index] > 0
                        },
                    },
                }
            )
        object_stats = _stats(samples["object_mass"])
        object_result: dict[str, Any] = {"mass_kg": object_stats}
        if scene.mass_kg is not None:
            object_result["absolute_error_kg"] = abs(object_stats["mean"] - scene.mass_kg)
        result = {
            "schema_version": 1,
            "method": "ABO500-MeshPhysics-Interior",
            "scene_id": scene.scene_id,
            "split": scene.split,
            "mesh": {
                "source": str(scene.mesh_path),
                "mesh_unit_to_meter": scene.mesh_unit_to_meter,
                "watertight": bool(mesh.is_watertight),
                "volume_m3": float(sum(part_volumes.values())),
                "repair": repair_report,
            },
            "object": object_result,
            "num_views": len(scene.views),
            "num_posterior_samples": samples["num_samples"],
            "parts": parts,
        }
        if scene.mass_kg is not None:
            result["target_mass_kg"] = scene.mass_kg
        return result

    def _feature_row(
        self,
        scene: ABOScene,
        part_volumes: dict[int, float],
        part_areas: dict[int, float],
        evidence: dict[int, MaterialEvidence],
        proposals: dict[int, PartInteriorProposal],
    ) -> dict[str, Any]:
        default_temperature = float(self.cfg.get("materials", {}).get("temperature", 1.0))
        expected_volume_by_material = np.zeros(len(self.material_names), dtype=np.float64)
        for part_id, part_volume in part_volumes.items():
            probabilities = calibrated_probabilities(
                evidence[part_id].mean_logits,
                self.material_names,
                None,
                default_temperature,
            )
            probabilities = self._fuse_surface_probabilities(
                probabilities, proposals[part_id]
            )
            expected = expected_material_volumes(
                part_volume,
                part_areas[part_id],
                probabilities,
                proposals[part_id],
                self.material_names,
            )
            expected_volume_by_material += np.asarray(
                [expected[name] for name in self.material_names], dtype=np.float64
            )
        result = {
            "scene_id": scene.scene_id,
            "split": scene.split,
            "material_names": self.material_names,
            "expected_material_volumes_m3": expected_volume_by_material.tolist(),
            "parts": [
                {
                    "part_id": part_id,
                    "volume_m3": float(part_volumes[part_id]),
                    "surface_area_m2": float(part_areas[part_id]),
                    "mean_logits": evidence[part_id].mean_logits.tolist(),
                    "num_views": len(evidence[part_id].view_logits),
                    "interior_proposal": proposals[part_id].to_json(),
                }
                for part_id in sorted(part_volumes)
            ],
        }
        if scene.mass_kg is not None:
            result["target_mass_kg"] = scene.mass_kg
        return result

    def _save_artifacts(
        self,
        output_dir: Path,
        mesh: trimesh.Trimesh,
        scene: ABOScene,
        segmentation: PartSegmentation,
        observations: dict[int, list],
        prediction: dict[str, Any],
    ) -> None:
        metric_mesh = mesh.copy()
        metric_mesh.apply_scale(scene.mesh_unit_to_meter)
        metric_mesh.export(output_dir / "mesh_metric.ply")
        segmented = metric_mesh.copy()
        palette = np.asarray(
            [
                [0, 114, 178, 255],
                [230, 159, 0, 255],
                [0, 158, 115, 255],
                [204, 121, 167, 255],
                [213, 94, 0, 255],
                [86, 180, 233, 255],
                [240, 228, 66, 255],
            ],
            dtype=np.uint8,
        )
        segmented.visual.face_colors = palette[segmentation.face_labels % len(palette)]
        segmented.export(output_dir / "mesh_parts.ply")
        np.save(output_dir / "face_labels.npy", segmentation.face_labels)
        if self.cfg.get("output", {}).get("save_crops", False):
            crop_dir = ensure_dir(output_dir / "part_crops")
            for part_id, part_observations in observations.items():
                for view_id, observation in enumerate(part_observations):
                    observation.crop.save(crop_dir / f"part_{part_id:02d}_view_{view_id:02d}.jpg")
        write_json(
            output_dir / "interior_proposals.json",
            [part["interior"]["proposal"] for part in prediction["parts"]],
        )
        write_json(output_dir / "prediction.json", prediction)


def run_abo500_split(
    dataset: ABO500Dataset,
    cfg: dict[str, Any],
    split: str,
    output_dir: str | Path,
    calibration_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    pipeline = ABO500MeshPipeline(cfg, calibration_path)
    scene_ids = dataset.scene_ids(split)
    if limit is not None:
        scene_ids = scene_ids[:limit]
    predictions: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for scene_id in tqdm(scene_ids, desc=f"ABO-500 {split}"):
        try:
            scene = dataset.get_scene(scene_id, split)
            prediction, feature = pipeline.infer_scene(scene, output_dir / "scenes" / scene_id)
            predictions.append(prediction)
            features.append(feature)
        except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append({"scene_id": scene_id, "error": str(exc)})
    write_json(output_dir / "predictions.json", predictions)
    write_jsonl(output_dir / "features.jsonl", features)
    summary = {
        "split": split,
        "num_requested": len(scene_ids),
        "num_succeeded": len(predictions),
        "num_failed": len(failures),
        "failures": failures,
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary


def _load_mesh(path: Path) -> trimesh.Trimesh:
    return load_triangle_mesh(path)


def _stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "q05": float(np.quantile(array, 0.05)),
        "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _unit_lognormal(rng: np.random.Generator, log_sigma: float) -> float:
    if log_sigma <= 0:
        return 1.0
    return float(rng.lognormal(-0.5 * log_sigma**2, log_sigma))


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - float(np.max(values))
    probability = np.exp(shifted)
    return probability / probability.sum()

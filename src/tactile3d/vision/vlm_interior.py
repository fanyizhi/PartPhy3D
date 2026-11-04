from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tactile3d.physics.interior import (
    ROLES,
    STRUCTURES,
    InteriorComponent,
    InteriorHypothesis,
    Interval,
    PartInteriorProposal,
    solid_proposal,
)
from tactile3d.vision.material_encoder import MaterialEvidence
from tactile3d.vision.multiview import ViewObservation

_ALIASES = {
    "abs": "plastic",
    "abs_plastic": "plastic",
    "polypropylene": "plastic",
    "polyethylene": "plastic",
    "polycarbonate": "plastic",
    "polyurethane_foam": "foam",
    "memory_foam": "foam",
    "cotton_batting": "fabric",
    "polyester_batting": "fabric",
    "cardboard": "paper_cardboard",
    "paper": "paper_cardboard",
    "iron": "steel",
    "metal": "steel",
}


class InteriorVLMEstimator:
    """Generate structured surface-material and latent-interior hypotheses.

    The OpenAI-compatible backend works with local vLLM servers as well as
    hosted endpoints. The file backend makes expensive VLM calls cacheable and
    keeps benchmark inference deterministic.
    """

    def __init__(self, material_names: list[str] | tuple[str, ...], cfg: dict[str, Any]) -> None:
        self.material_names = tuple(material_names)
        self.cfg = cfg
        self.backend = str(cfg.get("backend", "surface_only"))

    def propose(
        self,
        scene_id: str,
        part_id: int,
        observations: list[ViewObservation],
        evidence: MaterialEvidence,
        geometry: dict[str, float],
    ) -> PartInteriorProposal:
        surface_probabilities = _softmax(evidence.mean_logits)
        if self.backend in {"none", "surface_only"}:
            return solid_proposal(part_id, self.material_names, surface_probabilities)
        if self.backend == "heuristic":
            return self._heuristic(part_id, surface_probabilities)
        if self.backend == "file":
            payload = self._load_file(scene_id, part_id)
        elif self.backend == "openai_compatible":
            payload = self._request_vlm(scene_id, part_id, observations, geometry)
        else:
            raise ValueError(f"Unknown interior VLM backend: {self.backend}")
        return parse_part_proposal(
            payload,
            part_id,
            self.material_names,
            surface_probabilities,
            source=self.backend,
        )

    def _load_file(self, scene_id: str, part_id: int) -> dict[str, Any]:
        root = Path(str(self.cfg.get("response_root", "vlm_responses"))).expanduser()
        part_path = root / scene_id / f"part_{part_id:02d}.json"
        generated_path = root / scene_id / "interior_proposals.json"
        scene_path = root / f"{scene_id}.json"
        if part_path.is_file():
            return json.loads(part_path.read_text(encoding="utf-8"))
        if scene_path.is_file():
            payload = json.loads(scene_path.read_text(encoding="utf-8"))
            for part in payload.get("parts", []):
                if int(part.get("part_id", -1)) == part_id:
                    return part
        if generated_path.is_file():
            payload = json.loads(generated_path.read_text(encoding="utf-8"))
            for part in payload:
                if int(part.get("part_id", -1)) == part_id:
                    return part
        raise FileNotFoundError(
            f"No cached VLM response for {scene_id} part {part_id} under {root}"
        )

    def _request_vlm(
        self,
        scene_id: str,
        part_id: int,
        observations: list[ViewObservation],
        geometry: dict[str, float],
    ) -> dict[str, Any]:
        base_url = str(
            self.cfg.get("base_url")
            or os.environ.get("TACTILE3D_VLM_BASE_URL", "http://127.0.0.1:8000/v1")
        ).rstrip("/")
        endpoint = base_url if base_url.endswith("chat/completions") else f"{base_url}/chat/completions"
        model = str(self.cfg.get("model") or os.environ.get("TACTILE3D_VLM_MODEL", "Qwen3-VL-8B-Instruct"))
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _prompt(scene_id, part_id, geometry, self.material_names),
            }
        ]
        max_images = max(1, int(self.cfg.get("max_images_per_part", 4)))
        crop_limit = max_images
        if observations and max_images > 1:
            with Image.open(observations[0].image_path) as context_image:
                context_url = _image_data_url(context_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": context_url},
                }
            )
            crop_limit -= 1
        for observation in observations[:crop_limit]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(observation.crop)},
                }
            )
        request_payload: dict[str, Any] = {
            "model": model,
            "temperature": float(self.cfg.get("temperature", 0.15)),
            "max_tokens": int(self.cfg.get("max_tokens", 1800)),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a physical-material analyst. Return strict JSON only. "
                        "Visible surface material and hidden interior material are different variables."
                    ),
                },
                {"role": "user", "content": content},
            ],
        }
        if bool(self.cfg.get("json_response_format", True)):
            request_payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        api_key = str(self.cfg.get("api_key") or os.environ.get("TACTILE3D_VLM_API_KEY", ""))
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        retries = max(1, int(self.cfg.get("retries", 3)))
        timeout = float(self.cfg.get("timeout_seconds", 120))
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content_value = body["choices"][0]["message"]["content"]
                return _parse_response_content(content_value)
            except (KeyError, json.JSONDecodeError, urllib.error.URLError) as exc:
                if attempt + 1 == retries:
                    raise RuntimeError(
                        f"VLM request failed for {scene_id} part {part_id}: {exc}"
                    ) from exc
                time.sleep(float(self.cfg.get("retry_delay_seconds", 2.0)) * (attempt + 1))
        raise RuntimeError("Unreachable VLM retry state")

    def _heuristic(
        self, part_id: int, surface_probabilities: np.ndarray
    ) -> PartInteriorProposal:
        top_material = self.material_names[int(np.argmax(surface_probabilities))]
        probabilities = {
            name: float(value)
            for name, value in zip(self.material_names, surface_probabilities, strict=True)
        }
        solid = InteriorHypothesis(
            name="uniform solid",
            structure="solid",
            probability=1.0,
            wall_thickness_m=Interval(0.0, 0.0, 0.0),
            components=(),
        )
        if top_material in {"leather", "fabric"} and "foam" in self.material_names:
            hypotheses = (
                InteriorHypothesis(
                    name="upholstery over foam core",
                    structure="layered",
                    probability=0.75,
                    wall_thickness_m=Interval(0.0005, 0.0015, 0.004),
                    components=(
                        _component(top_material, "shell", 1.0, 1.0),
                        _component("foam", "core", 0.9, 1.0),
                        _component("wood", "frame", 0.1, 1.0),
                    ),
                ),
                InteriorHypothesis(
                    name="surface-only flexible shell",
                    structure="hollow",
                    probability=0.15,
                    wall_thickness_m=Interval(0.0005, 0.0015, 0.004),
                    components=(_component(top_material, "shell", 1.0, 1.0),),
                ),
                InteriorHypothesis(
                    name=solid.name,
                    structure=solid.structure,
                    probability=0.10,
                    wall_thickness_m=solid.wall_thickness_m,
                    components=solid.components,
                ),
            )
        elif top_material in {
            "plastic",
            "rubber",
            "silicone",
            "glass",
            "ceramic",
            "paper_cardboard",
            "aluminum",
            "steel",
        }:
            hypotheses = (
                InteriorHypothesis(
                    name="manufactured hollow shell",
                    structure="hollow",
                    probability=0.6,
                    wall_thickness_m=Interval(0.0008, 0.0025, 0.008),
                    components=(_component(top_material, "shell", 1.0, 1.0),),
                ),
                InteriorHypothesis(
                    name=solid.name,
                    structure=solid.structure,
                    probability=0.4,
                    wall_thickness_m=solid.wall_thickness_m,
                    components=solid.components,
                ),
            )
        else:
            hypotheses = (solid,)
        return PartInteriorProposal(
            part_id=part_id,
            object_category="unknown",
            source="heuristic",
            surface_material_probabilities=probabilities,
            hypotheses=hypotheses,
        )


def parse_part_proposal(
    payload: dict[str, Any],
    part_id: int,
    material_names: list[str] | tuple[str, ...],
    fallback_surface_probabilities: np.ndarray,
    source: str,
) -> PartInteriorProposal:
    allowed = tuple(material_names)
    surface = _parse_surface_probabilities(payload, allowed, fallback_surface_probabilities)
    hypotheses: list[InteriorHypothesis] = []
    for index, raw in enumerate(payload.get("interior_hypotheses", [])):
        structure = str(raw.get("structure", "solid")).strip().lower()
        if structure not in STRUCTURES:
            continue
        components = []
        for component_raw in raw.get("components", []):
            material = _material_name(component_raw.get("material", ""), allowed)
            role = str(component_raw.get("role", "core")).strip().lower()
            if role not in ROLES or (material is None and role != "void"):
                continue
            components.append(
                InteriorComponent(
                    material=material or "void",
                    role=role,
                    volume_fraction=_interval(
                        component_raw.get("volume_fraction", 1.0), 0.0, 1.0
                    ),
                    fill_ratio=_interval(component_raw.get("fill_ratio", 1.0), 0.0, 1.0),
                )
            )
        thickness_raw = raw.get("wall_thickness_mm", raw.get("wall_thickness_m", 0.0))
        multiplier = 0.001 if "wall_thickness_mm" in raw else 1.0
        thickness = _scale_interval(_interval(thickness_raw, 0.0, 1e4), multiplier)
        hypotheses.append(
            InteriorHypothesis(
                name=str(raw.get("name", f"hypothesis {index}")),
                structure=structure,
                probability=max(float(raw.get("probability", 0.0)), 0.0),
                wall_thickness_m=thickness,
                components=tuple(components),
            )
        )
    if not hypotheses:
        return solid_proposal(part_id, allowed, fallback_surface_probabilities, source=source)
    probabilities = _normalize(np.asarray([item.probability for item in hypotheses]))
    hypotheses = [
        InteriorHypothesis(
            name=item.name,
            structure=item.structure,
            probability=float(probability),
            wall_thickness_m=item.wall_thickness_m,
            components=item.components,
        )
        for item, probability in zip(hypotheses, probabilities, strict=True)
    ]
    return PartInteriorProposal(
        part_id=part_id,
        object_category=str(payload.get("object_category", "unknown")),
        source=source,
        surface_material_probabilities=surface,
        hypotheses=tuple(hypotheses),
    )


def _parse_surface_probabilities(
    payload: dict[str, Any],
    material_names: tuple[str, ...],
    fallback: np.ndarray,
) -> dict[str, float]:
    values = np.zeros(len(material_names), dtype=np.float64)
    raw = payload.get("surface_material_probabilities", payload.get("surface_materials", {}))
    if isinstance(raw, dict):
        entries = raw.items()
    elif isinstance(raw, list):
        entries = [
            (entry.get("material", entry.get("name", "")), entry.get("probability", 0.0))
            for entry in raw
            if isinstance(entry, dict)
        ]
    else:
        entries = []
    for name, probability in entries:
        normalized_name = _material_name(name, material_names)
        if normalized_name is not None:
            values[material_names.index(normalized_name)] += max(float(probability), 0.0)
    if values.sum() <= 0:
        values = np.asarray(fallback, dtype=np.float64)
    values = _normalize(values)
    return {name: float(values[index]) for index, name in enumerate(material_names)}


def _material_name(value: Any, material_names: tuple[str, ...]) -> str | None:
    name = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    name = _ALIASES.get(name, name)
    return name if name in material_names else None


def _interval(value: Any, minimum: float, maximum: float) -> Interval:
    if isinstance(value, dict):
        low = float(value.get("low", value.get("min", value.get("mode", 0.0))))
        high = float(value.get("high", value.get("max", value.get("mode", low))))
        mode = float(value.get("mode", value.get("mean", 0.5 * (low + high))))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        low, high = float(value[0]), float(value[-1])
        mode = float(value[1]) if len(value) >= 3 else 0.5 * (low + high)
    else:
        low = mode = high = float(value)
    low, high = sorted((np.clip(low, minimum, maximum), np.clip(high, minimum, maximum)))
    mode = float(np.clip(mode, low, high))
    return Interval(float(low), mode, float(high))


def _scale_interval(value: Interval, scale: float) -> Interval:
    return Interval(value.low * scale, value.mode * scale, value.high * scale)


def _component(material: str, role: str, volume_fraction: float, fill_ratio: float) -> InteriorComponent:
    return InteriorComponent(
        material=material,
        role=role,
        volume_fraction=Interval(volume_fraction, volume_fraction, volume_fraction),
        fill_ratio=Interval(fill_ratio, fill_ratio, fill_ratio),
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64) - float(np.max(values))
    return _normalize(np.exp(shifted))


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(values.sum())
    return values / total if total > 0 else np.full(len(values), 1.0 / len(values))


def _image_data_url(image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_response_content(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        text = "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    else:
        text = str(content)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("VLM response must be a JSON object")
    return payload


def _prompt(
    scene_id: str,
    part_id: int,
    geometry: dict[str, float],
    material_names: tuple[str, ...],
) -> str:
    return f"""
Analyze the highlighted/cropped object part from several views. Infer visible surface materials and
plausible hidden construction. Do not assume that upholstery, paint, leather, or fabric fills the
interior. Consider manufacturing patterns such as shells, foam cores, frames, infill, and voids.

Scene: {scene_id}
Part: {part_id}
Metric geometry: {json.dumps(geometry, sort_keys=True)}
Allowed material names: {json.dumps(material_names)}
Allowed structures: {json.dumps(STRUCTURES)}
Allowed component roles: {json.dumps(ROLES)}

Return one JSON object with this schema:
{{
  "object_category": "short category",
  "surface_materials": [{{"material": "allowed name", "probability": 0.0}}],
  "interior_hypotheses": [
    {{
      "name": "short construction description",
      "structure": "solid|hollow|layered|foam_filled|frame_supported|partial_infill",
      "probability": 0.0,
      "wall_thickness_mm": {{"low": 0.0, "mode": 0.0, "high": 0.0}},
      "components": [
        {{
          "material": "allowed name or void",
          "role": "shell|core|frame|infill|void",
          "volume_fraction": {{"low": 0.0, "mode": 0.0, "high": 0.0}},
          "fill_ratio": {{"low": 0.0, "mode": 0.0, "high": 0.0}}
        }}
      ]
    }}
  ]
}}

Probabilities at each level must sum to one. Component volume_fraction is the fraction of the
available shell or inner-core region. Use bulk density semantics for foam, fabric, and cardboard,
so their microscopic porosity must not be subtracted again. Produce 2-4 distinct hypotheses.
""".strip()

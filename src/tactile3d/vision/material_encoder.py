from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tactile3d.vision.multiview import ViewObservation


@dataclass(frozen=True)
class MaterialEvidence:
    material_names: tuple[str, ...]
    view_logits: np.ndarray
    view_weights: np.ndarray

    @property
    def mean_logits(self) -> np.ndarray:
        weights = self.view_weights / (self.view_weights.sum() + 1e-12)
        return (self.view_logits * weights[:, None]).sum(axis=0)


class MaterialEncoder:
    def __init__(self, material_names: Sequence[str], cfg: dict) -> None:
        self.material_names = tuple(material_names)
        self.cfg = cfg
        self.backend = str(cfg.get("backend", "handcrafted"))
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features = None

    def encode(self, observations: list[ViewObservation]) -> MaterialEvidence:
        if not observations:
            raise ValueError("At least one view observation is required")
        if self.backend == "open_clip":
            logits = self._open_clip_logits(observations)
        elif self.backend == "handcrafted":
            logits = np.stack([self._handcrafted_logits(item) for item in observations])
        else:
            raise ValueError(f"Unknown material feature backend: {self.backend}")
        weights = np.asarray([max(item.quality, 1e-6) for item in observations], dtype=np.float64)
        weights = np.sqrt(weights / weights.max())
        return MaterialEvidence(self.material_names, logits.astype(np.float64), weights)

    def _open_clip_logits(self, observations: list[ViewObservation]) -> np.ndarray:
        self._ensure_open_clip()
        import torch

        assert self._model is not None
        assert self._preprocess is not None
        images = torch.stack([self._preprocess(item.crop) for item in observations]).to(self._device)
        batch_size = int(self.cfg.get("batch_size", 16))
        chunks = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                features = self._model.encode_image(images[start : start + batch_size])
                chunks.append(features / features.norm(dim=-1, keepdim=True))
        image_features = torch.cat(chunks, dim=0)
        scale = float(self.cfg.get("similarity_scale", 30.0))
        return (scale * image_features @ self._text_features.T).float().cpu().numpy()

    def _ensure_open_clip(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP backend requested. Install the vision extra with "
                "pip install -e '.[vision]'."
            ) from exc
        model_name = str(self.cfg.get("model", "ViT-B-32"))
        pretrained = str(self.cfg.get("pretrained", "laion2b_s34b_b79k"))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self._device,
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        templates = self.cfg.get(
            "prompt_templates",
            [
                "a close-up photograph of an object part made of {}",
                "the surface material is {}",
            ],
        )
        prompts = [template.format(name.replace("_", " ")) for name in self.material_names for template in templates]
        tokens = tokenizer(prompts).to(self._device)
        with torch.no_grad():
            encoded = model.encode_text(tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            encoded = encoded.reshape(len(self.material_names), len(templates), -1).mean(dim=1)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        self._model = model.eval()
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._text_features = encoded

    def _handcrafted_logits(self, item: ViewObservation) -> np.ndarray:
        rgb = item.mean_rgb
        brightness = float(rgb.mean())
        neutral = 1.0 - float(rgb.max() - rgb.min())
        texture = min(item.texture * 8.0, 1.0)
        saturation = item.saturation
        scores: dict[str, float] = {
            "steel": 1.4 * neutral + 0.8 * brightness - 0.8 * texture,
            "aluminum": 1.2 * neutral + 0.9 * brightness - 0.5 * texture,
            "glass": 1.0 * brightness + 0.7 * neutral - 0.4 * saturation,
            "ceramic": 0.9 * brightness + 0.5 * neutral + 0.2 * texture,
            "plastic": 0.5 * saturation + 0.5 * (1.0 - texture),
            "rubber": 0.8 * (1.0 - brightness) + 0.4 * texture,
            "silicone": 0.5 * saturation + 0.4 * texture + 0.2 * brightness,
            "foam": 1.2 * texture + 0.5 * (1.0 - brightness),
            "wood": 0.8 * texture + 0.5 * rgb[0] - 0.2 * rgb[2],
            "paper_cardboard": 0.8 * texture + 0.5 * neutral + 0.2 * brightness,
            "fabric": 1.1 * texture + 0.4 * saturation,
            "leather": 0.6 * texture + 0.5 * (1.0 - brightness) + 0.3 * rgb[0],
            "stone": 0.9 * texture + 0.8 * neutral,
        }
        return np.asarray([scores.get(name, 0.0) for name in self.material_names], dtype=np.float64)

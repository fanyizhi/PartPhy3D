"""本地模型入口的标签隔离和非法输出验收，不下载或调用真实模型。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tactile3d.experiments import local_vlm_pilot
from tactile3d.vision import local_vlm
from tactile3d.vision.local_vlm import VLMResponse
from tactile3d.vision.material_encoder import MaterialEvidence
from tactile3d.vision.multiview import ViewObservation
from tactile3d.vision.vlm_interior import InteriorVLMEstimator


def response_payload() -> dict[str, Any]:
    return {
        "mass_kg": {"estimate": 2.0, "plausible_low": 1.0, "plausible_high": 3.0},
        "parts": [
            {
                "part_id": "p0",
                "name": "body",
                "surface_material": "plastic",
                "interior_material": "unknown",
                "bbox_first_view_1000": [100, 100, 900, 900],
            }
        ],
    }


def test_multimodal_messages_accept_system_text_and_transparency() -> None:
    transparent = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    messages = local_vlm.make_messages("analyze", [transparent], "system")
    assert all(isinstance(message["content"], list) for message in messages)
    assert messages[0]["content"][0] == {"type": "text", "text": "system"}
    image = messages[1]["content"][0]["image"]
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_unmapped_material_is_retained_without_assigning_physics() -> None:
    payload = response_payload()
    payload["parts"][0]["surface_material"] = "felt"
    result = local_vlm_pilot.validate_response(json.dumps(payload), {"plastic"})
    flags = local_vlm_pilot.quality_flags(result, {"plastic"})
    assert result["parts"][0]["surface_material"] == "felt"
    assert flags["unmapped_materials"] == ["felt"]


def test_qwen_fast_pixel_budget_updates_effective_size() -> None:
    from types import SimpleNamespace

    class QwenImageProcessorStub:
        def __init__(self) -> None:
            self.size = {"shortest_edge": 65536, "longest_edge": 16777216}

    processor = SimpleNamespace(image_processor=QwenImageProcessorStub())
    result = local_vlm.configure_qwen_pixels(processor, 196608, 12544)
    assert processor.image_processor.size == {
        "shortest_edge": 12544,
        "longest_edge": 196608,
    }
    assert result == {"min_pixels": 12544, "max_pixels": 196608}


@pytest.mark.parametrize("case", ["negative", "nan", "interval", "bbox", "duplicate"])
def test_reject_invalid_model_payload(case: str) -> None:
    payload = response_payload()
    if case == "negative":
        payload["mass_kg"]["estimate"] = -1
    elif case == "nan":
        payload["mass_kg"]["estimate"] = float("nan")
    elif case == "interval":
        payload["mass_kg"]["plausible_low"] = 10
    elif case == "bbox":
        payload["parts"][0]["bbox_first_view_1000"][0] = -2
    else:
        payload["parts"].append(dict(payload["parts"][0]))
    with pytest.raises(ValueError):
        local_vlm_pilot.validate_response(json.dumps(payload), {"plastic"})


def test_mass_labels_are_read_after_prediction_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"object_ids": ["sample"]}))
    root = tmp_path / "data"
    images = root / "scenes/sample/images"
    images.mkdir(parents=True)
    for view in local_vlm_pilot.VIEWS:
        Image.new("RGB", (32, 32), "white").save(images / f"sample_{view:02d}.png")
    weights = root / "filtered_product_weights.json"
    weights.write_text(json.dumps({"sample": 4.0}))
    output = tmp_path / "result"
    original_read = Path.read_text
    events = []

    def observed_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == weights:
            assert (output / "prediction_seal.json").is_file()
            events.append("labels")
        return original_read(path, *args, **kwargs)

    class FakeModel:
        def __init__(self, path: Path) -> None:
            self.provenance = {"source": "test_double_not_real_model"}

        def generate(self, *args: Any) -> VLMResponse:
            assert not events
            events.append("inference")
            return VLMResponse(json.dumps(response_payload()), 0.1, 10, 20, 0)

    monkeypatch.setattr(Path, "read_text", observed_read)
    monkeypatch.setattr(local_vlm_pilot, "LocalVLM", FakeModel)
    result = local_vlm_pilot.infer_pilot(tmp_path / "model", root, output, 1, cohort)
    assert events == ["inference", "labels"]
    assert result["successful"] == 1
    assert result["metrics_successful_objects_only"]["ade_kg"] == 2.0
    sealed = json.loads((output / "sealed_predictions.json").read_text())
    assert "target_mass_kg" not in sealed[0]


def test_quality_flags_preserve_conflicting_structure_hypothesis() -> None:
    payload = response_payload()
    payload["parts"][0]["construction"] = "solid"
    payload["parts"][0]["interior_material"] = "void"
    result = local_vlm_pilot.quality_flags(payload, {"plastic"})
    assert result["solid_with_void_interior"] == [payload["parts"][0]["part_id"]]
    assert payload["parts"][0]["interior_material"] == "void"


def test_local_interior_backend_reuses_one_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances, calls = [], []

    class FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            instances.append(self)

        def generate(
            self, prompt: str, images: list[Image.Image], system: str
        ) -> VLMResponse:
            calls.append(len(images))
            payload = {
                "surface_materials": [{"material": "plastic", "probability": 1.0}],
                "interior_hypotheses": [
                    {"name": "solid", "structure": "solid", "probability": 1.0}
                ],
            }
            return VLMResponse(json.dumps(payload), 0.1, 10, 20, 0)

    monkeypatch.setattr(local_vlm, "LocalVLM", FakeModel)
    image = Image.new("RGB", (32, 32), "white")
    path = tmp_path / "input.png"
    image.save(path)
    obs = ViewObservation(path, image, 1.0, 10, np.zeros(3), np.ones(3), 0, 0)
    evidence = MaterialEvidence(("plastic",), np.zeros((1, 1)), np.ones(1))
    estimator = InteriorVLMEstimator(
        ["plastic"],
        {
            "backend": "transformers_local",
            "model_path": str(tmp_path),
            "max_images_per_part": 1,
        },
    )
    for part in (0, 1):
        proposal = estimator.propose("sample", part, [obs], evidence, {})
        assert proposal.part_id == part
    assert len(instances) == 1
    assert calls == [1, 1]

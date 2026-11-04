from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tactile3d.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


def prepare_sft_dataset(
    annotations_path: str | Path,
    output_dir: str | Path,
    *,
    image_root: str | Path | None = None,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 17,
    max_images: int = 4,
) -> dict[str, Any]:
    """Convert expert annotations to a leakage-resistant multimodal SFT dataset.

    Splits are assigned from ``group_id`` or ``scene_id`` so all views and parts
    belonging to one physical object remain in the same partition.
    """

    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("train_fraction and val_fraction must lie in [0, 1]")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than one")
    if max_images < 1:
        raise ValueError("max_images must be positive")
    annotations_path = Path(annotations_path).expanduser().resolve()
    image_root_path = (
        Path(image_root).expanduser().resolve() if image_root else annotations_path.parent
    )
    output_dir = ensure_dir(Path(output_dir).expanduser().resolve())
    rows = read_jsonl(annotations_path)
    partitions: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    rejected: list[dict[str, str]] = []

    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", f"sample_{index:06d}"))
        try:
            sample = _convert_record(row, sample_id, image_root_path, max_images)
            split = str(row.get("split") or _split_for(row, train_fraction, val_fraction, seed))
            if split not in partitions:
                raise ValueError(f"unsupported split {split!r}")
            sample["split"] = split
            partitions[split].append(sample)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            rejected.append({"sample_id": sample_id, "error": str(exc)})

    for split, samples in partitions.items():
        write_jsonl(output_dir / f"{split}.jsonl", samples)
    report = {
        "schema_version": 1,
        "source": str(annotations_path),
        "split_unit": "physical_object",
        "counts": {split: len(samples) for split, samples in partitions.items()},
        "num_rejected": len(rejected),
        "rejected": rejected,
    }
    write_json(output_dir / "dataset_report.json", report)
    return report


def _convert_record(
    row: dict[str, Any], sample_id: str, image_root: Path, max_images: int
) -> dict[str, Any]:
    images_raw = row.get("images")
    if not isinstance(images_raw, list) or not images_raw:
        raise ValueError("images must be a non-empty list")
    images = []
    for raw in images_raw[:max_images]:
        path = Path(str(raw)).expanduser()
        path = path if path.is_absolute() else image_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"training image does not exist: {path}")
        images.append(str(path))

    target = row.get("expert_target")
    if not isinstance(target, dict) or not target:
        raise ValueError("expert_target must be a non-empty JSON object")
    geometry = row.get("geometry", {})
    evidence = row.get("professional_evidence", [])
    if not isinstance(geometry, dict) or not isinstance(evidence, list):
        raise TypeError("geometry must be an object and professional_evidence must be a list")

    prompt = {
        "task": (
            "Infer visible material and a probability distribution over hidden construction. "
            "Separate measured physical evidence from human visual expectation. Return JSON only."
        ),
        "scene_id": str(row.get("scene_id", sample_id)),
        "part_id": row.get("part_id"),
        "geometry": geometry,
        "professional_evidence": evidence,
        "output_schema": {
            "object_category": "string",
            "surface_material_probabilities": "mapping",
            "interior_hypotheses": "list",
            "human_expectation": "optional mapping",
            "evidence_ids": "list",
        },
    }
    user_content = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": json.dumps(prompt, sort_keys=True)})
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a calibrated physical-material analyst."}
            ],
        },
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": json.dumps(target, sort_keys=True)}],
        },
    ]
    return {
        "sample_id": sample_id,
        "group_id": str(row.get("group_id", row.get("scene_id", sample_id))),
        "images": images,
        "messages": messages,
        "provenance": row.get("provenance", {}),
    }


def _split_for(
    row: dict[str, Any], train_fraction: float, val_fraction: float, seed: int
) -> str:
    group = str(row.get("group_id", row.get("scene_id", row.get("sample_id", ""))))
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"

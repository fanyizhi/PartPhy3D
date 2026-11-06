"""训练入口的数据隔离与损失mask契约。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from tactile3d.training.collator import AssistantVisionCollator
from tactile3d.training.preflight import validate_training_files


def write_sample(path: Path, image: Path, group: str) -> None:
    row = {
        "group_id": group,
        "images": [str(image)],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "input"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_training_rejects_group_and_duplicate_image_leakage(tmp_path: Path) -> None:
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    Image.new("RGB", (16, 16), "red").save(first)
    Image.new("RGB", (16, 16), "blue").save(second)
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_sample(train, first, "same_object")
    write_sample(val, second, "same_object")
    with pytest.raises(ValueError, match="物体重叠"):
        validate_training_files(train, val)
    write_sample(val, first, "other_object")
    with pytest.raises(ValueError, match="相同图像"):
        validate_training_files(train, val)
    write_sample(val, second, "other_object")
    assert validate_training_files(train, val)["object_overlap"] is False


def test_assistant_collator_masks_prompt_and_padding() -> None:
    torch = pytest.importorskip("torch")

    class Processor:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(padding_side="left", pad_token_id=0)

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            prefix = "image_prompt:"
            return (
                prefix if add_generation_prompt else prefix + messages[-1]["text"] + "#"
            )

        def __call__(self, text: list[str], **kwargs: Any) -> dict[str, Any]:
            rows = [[ord(char) + 1 for char in value] for value in text]
            width = max(map(len, rows))
            return {
                "input_ids": torch.tensor([r + [0] * (width - len(r)) for r in rows]),
                "attention_mask": torch.tensor(
                    [[1] * len(r) + [0] * (width - len(r)) for r in rows]
                ),
            }

    processor = Processor()
    collator = AssistantVisionCollator(processor)
    examples = [
        {
            "images": ["placeholder"],
            "messages": [
                {"role": "user"},
                {"role": "assistant", "text": text},
            ],
        }
        for text in ("A", "longer answer")
    ]
    batch = collator(examples)
    labels = batch["labels"]
    assert torch.all(labels[:, : len("image_prompt:")] == -100)
    assert torch.all(labels[batch["attention_mask"] == 0] == -100)
    assert labels[0, len("image_prompt:")].item() == ord("A") + 1
    assert labels[0, len("image_prompt:") + 1].item() == ord("#") + 1
    assert processor.tokenizer.padding_side == "right"

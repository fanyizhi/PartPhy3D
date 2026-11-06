"""训练前验证数据文件、对象隔离和消息结构，不初始化GPU。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


def validate_training_files(train: Path, validation: Path) -> dict[str, Any]:
    """在下载或加载模型前检查训练和验证集，避免把付费GPU时间耗在坏输入上。"""
    groups: dict[str, set[str]] = {}
    image_hashes: dict[str, set[str]] = {}
    report: dict[str, Any] = {}
    for split, path in (("train", train), ("validation", validation)):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"训练数据不存在或为空：{path}")
        ids: set[str] = set()
        hashes: set[str] = set()
        count = 0
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"训练JSON无效：{path}:{number}") from error
                if not isinstance(row, dict):
                    raise TypeError("训练JSON记录必须是对象")
                group = row.get("group_id")
                if not isinstance(group, str) or not group.strip():
                    raise ValueError("每条训练记录必须有物理对象group_id")
                ids.add(group)
                images = row.get("images")
                if not isinstance(images, list) or not images:
                    raise ValueError("每条VLM训练记录需要非空images列表")
                for image in images:
                    if not isinstance(image, str) or not Path(image).is_file():
                        raise FileNotFoundError(f"训练图像不可访问：{image}")
                    with Image.open(image) as source:
                        source.verify()
                    hashes.add(hashlib.sha256(Path(image).read_bytes()).hexdigest())
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("训练记录缺少messages列表")
                if any(not isinstance(message, dict) for message in messages):
                    raise TypeError("训练消息必须是对象")
                if messages[-1].get("role") != "assistant":
                    raise ValueError("训练记录最后必须有assistant答案")
                for message in messages:
                    if not isinstance(message.get("content"), list):
                        raise TypeError("多模态消息content必须使用内容列表")
                count += 1
        if count == 0:
            raise ValueError(f"训练文件没有有效记录：{path}")
        groups[split] = ids
        image_hashes[split] = hashes
        report[split] = {
            "samples": count,
            "groups": len(ids),
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    overlap = groups["train"] & groups["validation"]
    if overlap:
        raise ValueError(f"训练与验证物体重叠：{sorted(overlap)}")
    if image_hashes["train"] & image_hashes["validation"]:
        raise ValueError("训练与验证出现相同图像字节，不能作为独立验证")
    return {**report, "split_unit": "physical_object", "object_overlap": False}

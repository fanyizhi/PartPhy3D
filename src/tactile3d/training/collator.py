"""VLM监督微调只在assistant答案上计算损失，保留完整图像token。"""

from __future__ import annotations

from typing import Any


class AssistantVisionCollator:
    """不依赖chat模板的generation标记，核对前缀后再屏蔽提示token。"""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not examples:
            raise ValueError("训练批次不能为空")
        full_text, prefix_text, images = [], [], []
        for example in examples:
            messages = example["messages"]
            if not messages or messages[-1]["role"] != "assistant":
                raise ValueError("训练记录最后一条消息必须为assistant答案")
            full_text.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            prefix_text.append(
                self.processor.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            images.append(example["images"])
        batch = self.processor(
            text=full_text,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        prefixes = self.processor(
            text=prefix_text,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for index in range(len(examples)):
            length = int(prefixes["attention_mask"][index].sum())
            if not torch.equal(
                batch["input_ids"][index, :length],
                prefixes["input_ids"][index, :length],
            ):
                raise ValueError("chat模板不能可靠定位assistant边界，拒绝训练错位标签")
            labels[index, :length] = -100
            if not torch.any(labels[index] != -100):
                raise ValueError("训练样本没有可监督的assistant token")
        batch["labels"] = labels
        return batch

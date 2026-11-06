"""本地多图VLM推理，显式记录模型、输入预算、耗时与GPU显存。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_MAX_TOKENS = 1800
DEFAULT_MAX_PIXELS = 1024 * 1024
DEFAULT_SEED = 17
MAX_IMAGES = 6


def configure_qwen_pixels(
    processor: Any, max_pixels: int, min_pixels: int | None = None
) -> dict[str, int]:
    """Fast版Qwen从size读取像素面积，仅设置max_pixels属性不会生效。"""
    if type(max_pixels) is not int or (
        min_pixels is not None and type(min_pixels) is not int
    ):
        raise TypeError("像素预算必须是整数")
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not type(image_processor).__name__.startswith("Qwen"):
        raise ValueError("当前像素预算适配器只支持Qwen图像processor")
    size = getattr(image_processor, "size", {})
    lower = min(int(size.get("shortest_edge", 65536)), max_pixels)
    if min_pixels is not None:
        lower = min_pixels
    if not 1024 <= lower <= max_pixels:
        raise ValueError("像素预算需要1024 <= min_pixels <= max_pixels")
    image_processor.min_pixels = lower
    image_processor.max_pixels = max_pixels
    image_processor.size = {**size, "shortest_edge": lower, "longest_edge": max_pixels}
    return {"min_pixels": lower, "max_pixels": max_pixels}


def make_messages(
    prompt: str, images: list[Image.Image], system: str
) -> list[dict[str, Any]]:
    """多模态processor逐项读取content，系统提示也必须使用内容列表。"""
    if not prompt.strip() or not system.strip():
        raise ValueError("模型提示不能为空")
    if not 1 <= len(images) <= MAX_IMAGES:
        raise ValueError(f"每次输入需要1到{MAX_IMAGES}张图")
    content: list[dict[str, Any]] = []
    for image in images:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
        content.append({"type": "image", "image": rgb})
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]


@dataclass(frozen=True)
class VLMResponse:
    text: str
    seconds: float
    input_tokens: int
    output_tokens: int
    peak_gpu_bytes: int
    preprocess_seconds: float = 0.0
    image_grid_thw: tuple[tuple[int, ...], ...] | None = None


class LocalVLM:
    """只加载明确指定的本地权重，不自动下载或回退为规则推断。

    Args:
        model_path: 已下载模型目录。
        max_tokens: 每次生成的最大token数。
        max_pixels: 每张图的像素预算，由模型processor执行缩放。
    """

    def __init__(
        self,
        model_path: Path,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        quantized: bool = False,
        adapter_path: Path | None = None,
    ) -> None:
        self.path = model_path.expanduser().resolve()
        if not (self.path / "config.json").is_file():
            raise FileNotFoundError(f"本地模型缺少config.json：{self.path}")
        if max_tokens <= 0 or max_pixels < 32 * 32:
            raise ValueError("token预算须为正，图像预算至少为1024像素")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except (ImportError, RuntimeError) as error:
            raise RuntimeError("本地VLM依赖导入失败，请检查统一requirements") from error
        if not torch.cuda.is_available():
            raise RuntimeError("本地VLM验证需要可用的CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前配置需要支持BF16的GPU")
        torch.manual_seed(DEFAULT_SEED)
        self.torch = torch
        self.max_tokens = max_tokens
        start = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            str(self.path), local_files_only=True, max_pixels=max_pixels
        )
        pixel_budget = configure_qwen_pixels(self.processor, max_pixels)
        quantization = None
        if quantized:
            from transformers import BitsAndBytesConfig

            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.path),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            quantization_config=quantization,
        ).eval()
        if adapter_path is not None:
            from peft import PeftModel

            if not (adapter_path / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(f"适配器不存在：{adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_path), is_trainable=False
            ).eval()
        torch.cuda.synchronize()
        self.provenance = {
            "path": str(self.path),
            "model_type": self.model.config.model_type,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "max_pixels": max_pixels,
            "effective_image_pixel_budget": pixel_budget,
            "max_new_tokens": max_tokens,
            "seed": DEFAULT_SEED,
            "quantization": "NF4_double_quant" if quantized else None,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "load_seconds": time.perf_counter() - start,
        }

    def generate(
        self, prompt: str, images: list[Image.Image], system: str
    ) -> VLMResponse:
        """同一模型复用多个对象，保留原始回答供独立审计。"""
        preprocessing_start = time.perf_counter()
        messages = make_messages(prompt, images, system)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        grid = inputs.get("image_grid_thw")
        image_grid = (
            tuple(tuple(int(v) for v in row) for row in grid.cpu().tolist())
            if grid is not None
            else None
        )
        self.torch.cuda.reset_peak_memory_stats()
        self.torch.cuda.synchronize()
        preprocess_seconds = time.perf_counter() - preprocessing_start
        start = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=False
            )
        self.torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        generated_only = generated[0, input_tokens:]
        text = self.processor.decode(generated_only, skip_special_tokens=True)
        return VLMResponse(
            text,
            seconds,
            input_tokens,
            int(generated_only.numel()),
            int(self.torch.cuda.max_memory_allocated()),
            preprocess_seconds,
            image_grid,
        )

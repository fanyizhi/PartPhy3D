from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from tactile3d.physics.part_io import write_json
from tactile3d.training.collator import AssistantVisionCollator
from tactile3d.training.preflight import validate_training_files
from tactile3d.vision.local_vlm import configure_qwen_pixels


def run_training(args: argparse.Namespace) -> None:
    """Run a reproducible single- or multi-GPU LoRA SFT experiment."""

    audit = validate_training_files(Path(args.train_data), Path(args.val_data))
    for name in ("batch_size", "gradient_accumulation", "eval_steps", "lora_rank"):
        if getattr(args, name) <= 0:
            raise ValueError(f"训练参数必须为正：{name}")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("学习率必须为有限正数")
    if not 1024 <= args.min_pixels <= args.max_pixels:
        raise ValueError("图像预算需要1024 <= min_pixels <= max_pixels")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("训练输出非空，请选择新目录或明确指定恢复检查点")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "training_data_audit.json", audit)

    try:
        import torch
        from datasets import Image as DatasetImage
        from datasets import Sequence, load_dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Install the training dependencies with: pip install -e '.[train]'"
        ) from exc

    quantization = None
    if args.qlora:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "quantization_config": quantization,
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.qlora:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires an NVIDIA CUDA GPU")
        device = max(local_rank, 0)
        torch.cuda.set_device(device)
        model_kwargs["device_map"] = {"": device}
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        **model_kwargs,
    )
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    configure_qwen_pixels(processor, args.max_pixels, args.min_pixels)
    processor.save_pretrained(args.output)
    dataset = load_dataset(
        "json", data_files={"train": args.train_data, "validation": args.val_data}
    )
    dataset = dataset.cast_column("images", Sequence(DatasetImage()))
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        tf32=True,
        max_length=None,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,
        seed=17,
        data_seed=17,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch_fused",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=processor,
        peft_config=peft_config,
        data_collator=AssistantVisionCollator(processor),
    )
    trained = trainer.train(resume_from_checkpoint=args.resume)
    if not math.isfinite(float(trained.metrics["train_loss"])):
        raise RuntimeError("训练损失不是有限数，不能报告训练成功")
    trainer.save_model(args.output)
    trainer.save_state()
    updated = sum(
        int(torch.count_nonzero(parameter).item() > 0)
        for name, parameter in model.named_parameters()
        if "lora_B" in name and parameter.requires_grad
    )
    write_json(
        output / "training_report.json",
        {
            "scope": args.run_scope,
            "optimizer_steps": trainer.state.global_step,
            "metrics": trained.metrics,
            "nonzero_lora_B_tensors": updated,
            "loss_supervision": "assistant_tokens_only",
            "qlora": bool(args.qlora),
            "real_material_accuracy_established": False,
        },
    )


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--run-scope", choices=("research", "integration_smoke"), default="research"
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=384 * 28 * 28)
    parser.add_argument(
        "--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa"
    )
    parser.add_argument("--resume")
    parser.add_argument("--report-to", default="none")

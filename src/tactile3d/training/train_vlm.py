from __future__ import annotations

import argparse
import os


def run_training(args: argparse.Namespace) -> None:
    """Run a reproducible single- or multi-GPU LoRA SFT experiment."""

    try:
        import torch
        from datasets import Image as DatasetImage
        from datasets import Sequence, load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies with: pip install -e '.[train]'") from exc

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
        logging_steps=10,
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
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
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
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--resume")
    parser.add_argument("--report-to", default="none")

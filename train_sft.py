#!/usr/bin/env python3
"""SFT warm-start for the office-document task agent.

Trains a small student model (default: Qwen/Qwen2.5-Coder-3B-Instruct) on
teacher trajectories collected from a stronger model (Kimi-K2.5) and filtered
by `data_pipeline/build_sft_corpus.py`.

Output is a LoRA adapter saved to `--output-dir`, optionally pushed to HF
Hub for use as the base of GRPO continued training.

Designed for HF Jobs (1× A100 80GB, ~$2.50/hr, ~6 hours = ~$15) but runs
locally too.

Hardware sizing:
  - 3B base + LoRA r=32 + bf16 + 8K context: ~24 GB VRAM
  - Fits comfortably on A100 40GB / L40S 48GB / A100 80GB
  - For OOM, drop --max-seq-len to 4096 or --lora-r to 16

Example:
    pip install -U "trl>=0.11" "peft>=0.13" "transformers>=4.46" \
        "datasets>=3.0" "accelerate>=1.0" "bitsandbytes>=0.43"

    python train_sft.py \
        --dataset data/sft_kimi_k25.jsonl \
        --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
        --output-dir checkpoints/qwen3b-sft-kimi \
        --epochs 2 --lora-r 32

HF Jobs:
    hf jobs run \
        --hardware "Nvidia A100 - large" \
        --timeout 8h \
        --image "huggingface/transformers-pytorch-gpu:latest" \
        --secrets HF_TOKEN \
        -- \
        bash -c "pip install -U trl peft accelerate bitsandbytes && \
                 python train_sft.py --dataset data/sft_kimi_k25.jsonl \
                                     --output-dir /tmp/qwen3b-sft \
                                     --push-to-hub bpHigh/qwen3b-office-sft"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="path to the SFT corpus JSONL (built by "
                        "data_pipeline/build_sft_corpus.py)")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    p.add_argument("--output-dir", default="checkpoints/qwen3b-sft")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="learning rate (LoRA defaults are higher than full FT)")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="all-linear",
                   help="LoRA target modules; 'all-linear' is the safe default")
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8,
                   help="effective batch = per_device_bsz × grad_accum × n_gpus")
    p.add_argument("--max-seq-len", type=int, default=8192,
                   help="drop to 4096 if OOM on smaller GPUs")
    p.add_argument("--logging-steps", type=int, default=2)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--use-qlora", action="store_true",
                   help="4-bit quantization (slower, much less memory)")
    p.add_argument("--no-assistant-only-loss", action="store_true",
                   help="disable assistant-only loss masking; train on full "
                        "conversation tokens (legacy behavior)")
    p.add_argument("--push-to-hub", default="",
                   help="HF Hub repo to push the LoRA adapter to "
                        "(e.g., 'username/repo-name'). Optional.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-to", default="none",
                   help="'none', 'wandb', 'tensorboard', or comma-separated")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()

    # Heavy imports inside main so --help is fast and import failures get
    # reported with context.
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # ---- 1. Dataset ----
    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"ERROR: dataset {ds_path} not found", file=sys.stderr)
        print("Run data_pipeline/build_sft_corpus.py first.", file=sys.stderr)
        return 1

    print(f"Loading SFT corpus from {ds_path}")
    raw = load_dataset("json", data_files=str(ds_path), split="train")
    print(f"  rows: {len(raw)}")
    print(f"  cols: {raw.column_names}")
    if "messages" not in raw.column_names:
        print(f"ERROR: dataset is missing 'messages' column", file=sys.stderr)
        return 1

    # SFTTrainer wants ONLY the messages column (extra cols are tolerated but
    # cleaner to drop).  Keep score/n_steps for inspection in logs.
    keep = [c for c in raw.column_names if c == "messages"]
    drop = [c for c in raw.column_names if c not in keep]
    train_ds = raw.remove_columns(drop) if drop else raw

    # ---- 2. Tokenizer ----
    print(f"\nLoading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 3. Model ----
    print(f"Loading base model: {args.base_model}")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="sdpa",
    )
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  using 4-bit QLoRA")

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False  # required for grad checkpointing

    # ---- 4. LoRA ----
    target = args.target_modules
    if target != "all-linear" and "," in target:
        target = [t.strip() for t in target.split(",")]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # ---- 5. Trainer config ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_len,
        # Assistant-only loss: only compute loss on assistant tokens, mask
        # everything else.  This is the right behavior for multi-turn agent
        # SFT — we don't want to train on tool-feedback (which the env
        # generates, not the model).
        assistant_only_loss=not args.no_assistant_only_loss,
        # Don't pack — multi-turn examples are long enough on their own
        packing=False,
        report_to=args.report_to.split(",") if args.report_to != "none" else "none",
        seed=args.seed,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub or None,
        hub_strategy="end",
        hub_private_repo=False,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    # ---- 6. Train ----
    print("\nStarting SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # ---- 7. Save ----
    print(f"\nSaving final LoRA adapter to {out_dir}")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Save the run args for reproducibility
    with open(out_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.push_to_hub:
        print(f"Pushing to HF Hub: {args.push_to_hub}")
        trainer.push_to_hub()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

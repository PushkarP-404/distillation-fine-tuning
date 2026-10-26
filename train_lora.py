#!/usr/bin/env python3
"""
train_lora.py (v2.0)
--------------------
Fine-tunes SmolLM2 (135M / 360M) or Qwen2.5 (0.5B / 1.5B) using LoRA / QLoRA
on the Harness v2.3 task planning dataset.

Upgraded to support:
- Expanded context window (max_length=1536 by default) for multi-step plans.
- Train / validation split with eval metrics per epoch.
- Configurable gradient accumulation steps and target model architecture.
- Hardware auto-detection (CUDA / ROCm / MPS / CPU).
"""

import os
import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for AI-Agent OS Task Planner")
    parser.add_argument("--dataset", default="planning_chatml.jsonl", help="ChatML dataset path")
    parser.add_argument("--output-dir", default="./adapters/os_planner_lora", help="Output directory for LoRA adapter")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct", help="Base HuggingFace model ID")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device training batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length (prevents plan truncation)")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split ratio")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset
    if not os.path.exists(dataset_path) and os.path.exists("distilled_chatml.jsonl"):
        dataset_path = "distilled_chatml.jsonl"

    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at {dataset_path}.")
        return

    # Load via HuggingFace Datasets
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Loaded {len(raw_dataset)} training examples.")

    # Hardware detection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[HARDWARE] GPU detected: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        torch_dtype = torch.float16
    else:
        print("[HARDWARE] No GPU detected. Running on CPU.")
        torch_dtype = torch.float32

    print(f"Loading tokenizer: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    def apply_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return example

    dataset = raw_dataset.map(apply_chat_template)

    if args.val_split > 0 and len(dataset) > 20:
        split_data = dataset.train_test_split(test_size=args.val_split, seed=42)
        train_data = split_data["train"]
        eval_data = split_data["test"]
        print(f"Dataset split: {len(train_data)} train, {len(eval_data)} validation.")
    else:
        train_data = dataset
        eval_data = None

    print(f"Loading base model: {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch_dtype
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    print("Configuring training arguments...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_data is not None else "no",
        optim="adamw_torch",
        remove_unused_columns=False,
        use_cpu=not torch.cuda.is_available(),
    )

    print("Initializing LoRA Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print(f"Starting fine-tuning (Epochs: {args.epochs}, Max Sequence: {args.max_length})...")
    trainer.train()

    print(f"\nTraining complete! Saving fine-tuned LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nNext steps:")
    print("1. Evaluate using: python eval_planning_agent.py --adapter " + args.output_dir)
    print("2. Export to GGUF using: python export_gguf.py --lora-dir " + args.output_dir + " --quantization Q4_K_M")


if __name__ == "__main__":
    main()

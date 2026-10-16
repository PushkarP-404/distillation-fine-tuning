import os
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for AI-Agent OS")
    parser.add_argument("--dataset", default="distilled_chatml.jsonl", help="ChatML dataset path")
    parser.add_argument("--output-dir", default="./adapters/os_agent_lora", help="Output directory for LoRA adapter")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct", help="Base HuggingFace model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_path = args.dataset
    if not os.path.exists(dataset_path) and os.path.exists("planning_chatml.jsonl"):
        dataset_path = "planning_chatml.jsonl"
    
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}.")
        return

    # Load via HF Datasets directly
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Loaded {len(dataset)} training examples.")

    print(f"Loading base model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    def apply_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return example

    dataset = dataset.map(apply_chat_template)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
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
        max_length=512,  # Compact length for Micro-CoT + JSON verdict
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=5,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        optim="adamw_torch",
        remove_unused_columns=False,
        use_cpu=not torch.cuda.is_available(),
    )

    print("Initializing LoRA Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting distillation fine-tuning...")
    trainer.train()

    print(f"Training complete! Saving distilled LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\nNext steps:")
    print("1. Evaluate using 03_eval_distilled_agent.py")
    print("2. Export to GGUF using export_gguf.py --quantization Q4_K_M")

if __name__ == "__main__":
    main()

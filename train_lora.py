import os
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Configuration
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_PATH = os.environ.get("DATASET_PATH", "distilled_chatml.jsonl")  # Distilled ChatML
OUTPUT_DIR = "./adapters/os_agent_lora"

def main():
    print(f"Loading distilled dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}.")
        print("Run 01_distill_groq.py and 02_format_dataset.py first.")
        return

    # Load via HF Datasets directly
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Loaded {len(dataset)} training examples.")

    print(f"Loading base model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # HuggingFace Datasets map using ChatML template
    def apply_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return example

    dataset = dataset.map(apply_chat_template)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
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
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=5,
        num_train_epochs=20, # Increased for small dataset
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

    print(f"Training complete! Saving distilled LoRA adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\nNext steps:")
    print("1. Test using 03_eval_distilled_agent.py")
    print("2. Export to GGUF using export_gguf.py")

if __name__ == "__main__":
    main()

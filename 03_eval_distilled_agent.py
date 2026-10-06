import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

def load_models(adapter_path):
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    print(f"Loading distilled adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer

def evaluate(model, tokenizer, test_file):
    print(f"Loading test set from {test_file}...")
    
    with open(test_file, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
        
    correct = 0
    total = min(len(data), 50) # Evaluate on subset
    
    print(f"Evaluating {total} samples...")
    for item in data[:total]:
        prompt = item["prompt"]
        expected = item.get("teacher_completion", "")
        
        # Build prompt using ChatML
        chat = [
            {"role": "system", "content": "You are the operating system's kernel AI agent."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False)
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Simple evaluation logic: check if the verdict action matches
        expected_verdict = '"action": "ALLOW"' if '"action": "ALLOW"' in expected else '"action": "DENY"'
        actual_verdict = '"action": "ALLOW"' if '"action": "ALLOW"' in response else '"action": "DENY"'
        
        if expected_verdict == actual_verdict:
            correct += 1
            
        print(f"Query ID: {item.get('query_id')}")
        print(f"Expected: {expected_verdict} | Actual: {actual_verdict}")
        print("-" * 50)
        
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\nFinal Accuracy (Verdict Exact Match): {acc:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="./adapters/distilled_lora", help="Path to distilled LoRA adapter")
    parser.add_argument("--test", default="distilled_raw.jsonl", help="Path to evaluation dataset")
    args = parser.parse_args()
    
    model, tokenizer = load_models(args.adapter)
    evaluate(model, tokenizer, args.test)

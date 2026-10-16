import os
import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.modules['torchvision'] = None
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

CANONICAL_SYSTEM_PROMPT = """You are an expert Linux kernel security AI intercepting syscall queries.
You must analyze the process query and return your verdict strictly as a single JSON object.
Do not include markdown fences, reasoning preambles, or conversational commentary.

Output Format Requirements:
1. "analysis": Exactly 1-2 concise sentences (strictly maximum 35 words) explaining the causal security policy.
2. "action": Strictly either "ALLOW" or "DENY".
3. "reason": A short category tag (maximum 6 words)."""

def extract_first_json_object(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == '\\':
            if in_string:
                escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        return None
    return None

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

def evaluate(model, tokenizer, test_file, num_samples=25):
    print(f"Loading test set from {test_file}...")
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return

    with open(test_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
        
    valid_json_count = 0
    bounded_cot_count = 0
    correct_action_count = 0
    total = min(len(data), num_samples)
    
    print(f"Evaluating {total} samples with Greedy Decoding (temp=0.0)...\n")
    for idx, item in enumerate(data[:total], 1):
        if "messages" in item:
            prompt = next(m["content"] for m in item["messages"] if m["role"] == "user")
            expected_raw = next(m["content"] for m in item["messages"] if m["role"] == "assistant")
        else:
            prompt = item["prompt"]
            expected_raw = item.get("teacher_completion", "")
        
        chat = [
            {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,  # Greedy decoding (temp=0.0)
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        expected_obj = extract_first_json_object(expected_raw) or {}
        actual_obj = extract_first_json_object(response)
        
        is_json = actual_obj is not None
        if is_json:
            valid_json_count += 1
            words = actual_obj.get("analysis", "").split()
            if 0 < len(words) <= 35:
                bounded_cot_count += 1
            
            exp_action = expected_obj.get("action", "").upper()
            act_action = actual_obj.get("action", "").upper()
            if exp_action and exp_action == act_action:
                correct_action_count += 1
        
        print(f"[{idx}/{total}] Query ID: {item.get('query_id')}")
        print(f"  Valid JSON: {'YES' if is_json else 'NO'}")
        if is_json:
            print(f"  Expected Action: {expected_obj.get('action')} | Actual Action: {actual_obj.get('action')}")
            print(f"  Analysis ({len(actual_obj.get('analysis','').split())} words): \"{actual_obj.get('analysis')}\"")
        else:
            print(f"  RAW NON-JSON RESPONSE: {repr(response[:150])}")
        print("-" * 60)
        
    print("\n==================================================")
    print("           EVALUATION BENCHMARK RESULTS           ")
    print("==================================================")
    print(f"Total Samples Evaluated:       {total}")
    print(f"JSON Validity Rate:           {(valid_json_count/total)*100:.1f}% ({valid_json_count}/{total})")
    print(f"Micro-CoT Compliance (<=35w): {(bounded_cot_count/total)*100:.1f}% ({bounded_cot_count}/{total})")
    print(f"Verdict Match Accuracy:        {(correct_action_count/total)*100:.1f}% ({correct_action_count}/{total})")
    print("==================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="./adapters/os_agent_lora", help="Path to distilled LoRA adapter")
    parser.add_argument("--test", default="distilled_chatml.jsonl", help="Path to evaluation dataset")
    parser.add_argument("--samples", type=int, default=25, help="Number of test samples to evaluate")
    args = parser.parse_args()
    
    model, tokenizer = load_models(args.adapter)
    evaluate(model, tokenizer, args.test, args.samples)

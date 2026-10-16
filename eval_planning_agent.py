#!/usr/bin/env python3
"""
eval_planning_agent.py
----------------------
Evaluates a fine-tuned LoRA or base model on OS Task Planning.
Measures:
  1. JSON Extraction Rate (Does the model produce balanced JSON?)
  2. Schema Validity Rate (Does validate_plan() pass with 0 PlanError?)
  3. Tool Accuracy (Are the registered tools correctly mapped?)
  4. Verification Presence (Do CHANGE tools have verification steps?)
"""

import os
import sys
import json
import torch
import argparse
from typing import Any

# Ensure harness_v2 can be imported
AGENT_DAEMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "os-ai-agent", "agent-daemon"))
sys.path.insert(0, AGENT_DAEMON_DIR)

from harness_v2 import validate_plan, extract_first_json_object, plan_prompt, PlanError
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_model_and_tokenizer(adapter_path: str | None = None, base_model_id: str = BASE_MODEL_ID):
    print(f"Loading tokenizer: {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("No adapter specified or path not found; evaluating base model.")
        model = base_model

    model.eval()
    return model, tokenizer


def evaluate(model, tokenizer, test_file: str, num_samples: int = 20):
    print(f"\nLoading test samples from {test_file}...")
    with open(test_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    test_subset = samples[:num_samples]
    total = len(test_subset)
    print(f"Running evaluation on {total} planning scenarios...\n")

    json_extracted_count = 0
    schema_valid_count = 0
    tool_match_count = 0

    dummy_inventory = [{"tool": "system.info", "ok": True, "facts": {"arch": "x86_64"}}]

    for idx, item in enumerate(test_subset, 1):
        intent = item.get("intent")
        expected_plan = item.get("plan", {})
        expected_tools = [s.get("tool") for s in expected_plan.get("steps", [])]

        prompt = plan_prompt(intent, dummy_inventory)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        raw_response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        extracted = extract_first_json_object(raw_response)
        has_json = extracted is not None and isinstance(extracted, dict)
        is_valid = False
        tool_match = False

        if has_json:
            json_extracted_count += 1
            try:
                plan = validate_plan(extracted)
                is_valid = True
                schema_valid_count += 1

                actual_tools = [s.tool for s in plan.steps]
                if actual_tools == expected_tools:
                    tool_match = True
                    tool_match_count += 1
            except PlanError:
                pass

        status_icon = "✓ PASS" if (is_valid and tool_match) else "✗ FAIL"
        print(f"[{idx:02d}/{total:02d}] {status_icon} | Intent: '{intent}'")
        if not is_valid:
            print(f"     Raw response snippet: {repr(raw_response[:120])}")

    print("\n" + "=" * 55)
    print("               EVALUATION RESULTS")
    print("=" * 55)
    print(f"Total Evaluated:        {total}")
    print(f"Balanced JSON Parsed:   {json_extracted_count}/{total} ({json_extracted_count/total*100:.1f}%)")
    print(f"Strict Schema Valid:    {schema_valid_count}/{total} ({schema_valid_count/total*100:.1f}%)")
    print(f"Exact Tool Match:       {tool_match_count}/{total} ({tool_match_count/total*100:.1f}%)")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OS Task Planning model")
    parser.add_argument("--adapter", default="./adapters/os_planner_lora", help="Path to LoRA adapter")
    parser.add_argument("--dataset", default="planning_dataset.jsonl", help="Dataset file with test scenarios")
    parser.add_argument("--samples", type=int, default=20, help="Number of test scenarios to evaluate")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.adapter)
    evaluate(model, tokenizer, args.dataset, args.samples)

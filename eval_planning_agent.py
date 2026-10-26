#!/usr/bin/env python3
"""
eval_planning_agent.py (v2.0)
-----------------------------
Evaluates a fine-tuned LoRA or base model on OS Task Planning.
Measures:
  1. JSON Extraction Rate (Does the model produce balanced JSON?)
  2. Schema Validity Rate (Does validate_plan() pass with 0 PlanError?)
  3. Tool Catalogue Adherence (Are only permitted tools used?)
  4. Verification Coverage (Do CHANGE steps include verification?)
  5. Multi-Step Execution DAG Consistency (Are depends_on links valid?)
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

from harness_v2 import (
    validate_plan, extract_first_json_object, plan_prompt, PlanError,
    _build_tool_subset, PolicyEngine, ALLOWED_TOOLS
)
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
        device_map="auto" if torch.cuda.is_available() else None,
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


def evaluate(model, tokenizer, test_file: str, num_samples: int = 25):
    print(f"\nLoading test samples from {test_file}...")
    with open(test_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    # Filter to plan types for schema validation
    plan_samples = [s for s in samples if s.get("type", "plan") == "plan"]
    test_subset = plan_samples[:num_samples] if plan_samples else samples[:num_samples]
    total = len(test_subset)
    print(f"Running evaluation on {total} planning scenarios (Harness v2.3 compliant)...\n")

    json_extracted_count = 0
    schema_valid_count = 0
    tool_allowed_count = 0
    verify_covered_count = 0
    multi_step_count = 0

    dummy_inventory = [{"tool": "system.info", "ok": True, "facts": {"arch": "x86_64"}}]

    for idx, item in enumerate(test_subset, 1):
        intent = item.get("intent", "")
        tool_subset = _build_tool_subset(intent)
        prompt = plan_prompt(intent, dummy_inventory, tool_subset=tool_subset)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        raw_response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        extracted = extract_first_json_object(raw_response)
        has_json = extracted is not None and isinstance(extracted, dict)
        is_valid = False
        all_tools_allowed = False
        verify_covered = False
        step_count = 0

        if has_json:
            json_extracted_count += 1
            try:
                plan = validate_plan(extracted)
                is_valid = True
                schema_valid_count += 1
                step_count = len(plan.steps)
                if step_count > 1:
                    multi_step_count += 1

                # Check if all used tools are in allowed list
                tools_used = [s.tool for s in plan.steps]
                all_tools_allowed = all(t in ALLOWED_TOOLS for t in tools_used)
                if all_tools_allowed:
                    tool_allowed_count += 1

                # Check verification on change steps
                change_steps = [s for s in plan.steps if s.tool in PolicyEngine.CHANGE]
                if all(len(s.verify) > 0 for s in change_steps):
                    verify_covered = True
                    verify_covered_count += 1

            except PlanError:
                pass

        status = "PASS" if is_valid else "FAIL"
        print(f"[{idx:02d}/{total:02d}] [{status}] (steps: {step_count}) | Intent: '{intent[:65]}'")
        if not is_valid:
            print(f"       Raw snippet: {repr(raw_response[:100])}")

    print("\n" + "=" * 55)
    print("               EVALUATION RESULTS")
    print("=" * 55)
    print(f"Total Evaluated:          {total}")
    print(f"JSON Objects Parsed:      {json_extracted_count}/{total} ({json_extracted_count/total*100:.1f}%)")
    print(f"Strict Schema Valid:      {schema_valid_count}/{total} ({schema_valid_count/total*100:.1f}%)")
    print(f"Allowed Tools Compliant:  {tool_allowed_count}/{total} ({tool_allowed_count/total*100:.1f}%)")
    print(f"Verification Enforced:    {verify_covered_count}/{total} ({verify_covered_count/total*100:.1f}%)")
    print(f"Multi-Step Plans:         {multi_step_count}/{total} ({multi_step_count/total*100:.1f}%)")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OS Task Planning model on Harness v2.3 benchmarks")
    parser.add_argument("--adapter", default="./adapters/os_planner_lora", help="Path to LoRA adapter")
    parser.add_argument("--dataset", default="planning_dataset.jsonl", help="Dataset file with test scenarios")
    parser.add_argument("--samples", type=int, default=25, help="Number of test scenarios to evaluate")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.adapter)
    evaluate(model, tokenizer, args.dataset, args.samples)

#!/usr/bin/env python3
"""
format_planning_dataset.py
--------------------------
Converts generated planning pairs into standard ChatML instruction dataset
for supervised fine-tuning (SFT) of SmolLM2-135M / 360M using HuggingFace / TRL.
"""

import os
import sys
import json
import argparse

AGENT_DAEMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "os-ai-agent", "agent-daemon"))
sys.path.insert(0, AGENT_DAEMON_DIR)

from harness_v2 import ALLOWED_TOOLS, validate_plan, extract_first_json_object

SYSTEM_PROMPT = (
    "You are the AI-Agent OS planner. Return one JSON object only. Do not return shell commands. "
    f"Use only the registered tools and ensure every system-changing step has verification. "
    f"Registered tools: {sorted(ALLOWED_TOOLS)}.\n"
    "Schema: {\"goal\":string,\"assumptions\":[string],\"success_criteria\":[object],"
    "\"steps\":[{\"id\":string,\"purpose\":string,\"tool\":string,\"args\":object,"
    "\"depends_on\":[string],\"verify\":[{\"tool\":string,\"args\":object}]}]}.\n"
    "End your JSON response with <|im_end|>."
)


def format_dataset(input_path: str, output_path: str):
    print(f"Formatting {input_path} -> {output_path}...")
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file {input_path} does not exist.")
        return

    chatml_entries = []
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            intent = item.get("intent")
            plan_obj = item.get("plan")

            if not intent or not plan_obj:
                skipped += 1
                continue

            # Ensure plan validates
            try:
                validate_plan(plan_obj)
            except Exception:
                skipped += 1
                continue

            assistant_content = json.dumps(plan_obj, indent=2, ensure_ascii=False)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Intent: {intent}"},
                {"role": "assistant", "content": assistant_content}
            ]

            chatml_entries.append({
                "id": item.get("id", line_num),
                "messages": messages
            })

    print(f"Validated and formatted {len(chatml_entries)} ChatML examples (skipped {skipped} invalid).")

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in chatml_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved ChatML training dataset to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format OS planning dataset into ChatML JSONL")
    parser.add_argument("--input", default="planning_dataset.jsonl", help="Input planning pairs JSONL")
    parser.add_argument("--output", default="planning_chatml.jsonl", help="Output ChatML JSONL")
    args = parser.parse_args()

    format_dataset(args.input, args.output)

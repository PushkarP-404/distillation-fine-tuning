#!/usr/bin/env python3
"""
format_planning_dataset.py (v2.0)
---------------------------------
Converts generated planning pairs into standard ChatML instruction dataset
for supervised fine-tuning (SFT) of SmolLM2-135M / 360M / Qwen2.5 using HuggingFace / TRL.

Upgraded to align with AI-Agent OS Harness v2.3:
- Category-filtered tool subsets dynamically generated per intent via _build_tool_subset().
- System prompt and schema matching harness_v2.plan_prompt() exactly.
- Multi-mode support:
  1. Standard & Multi-step Execution Plans
  2. High-level Task Chunking (Phase Decomposition)
  3. Step Error Recovery & Repair Traces
- Preserves ChatML structure compatible with tokenizer.apply_chat_template().
"""

import os
import sys
import json
import random
import argparse

AGENT_DAEMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "os-ai-agent", "agent-daemon"))
sys.path.insert(0, AGENT_DAEMON_DIR)

from harness_v2 import (
    ALLOWED_TOOLS, validate_plan, classify_intent_categories, _build_tool_subset
)

# Standard tool documentation matching harness_v2.plan_prompt
TOOL_TIPS = (
    "Tool tips:\n"
    "  file.list  {path, pattern?, recursive?, max_items?} – list directory contents\n"
    "  file.move  {src, dst} – move or rename a file\n"
    "  file.delete {path} – delete a single file\n"
    "  file.append {path, content} – append text to a file\n"
    "  shell.script {code, interpreter?, timeout?, cwd?, args?} – execute multi-line script\n"
    "  data.csv_read {path, delimiter?, max_rows?} – read CSV file into structured JSON\n"
    "  data.csv_write {path, headers?, rows, mode?} – write/append rows to CSV\n"
    "  document.read {path, max_bytes?} – extract text from .pdf, .docx, .txt, .md\n"
    "  email.send {to, subject, body, attachments?} – send or spool an email\n"
    "  email.read {folder?, max_emails?, search?, unseen_only?} – read emails from inbox/spool\n"
    "  http.get   {url, headers?, timeout?, max_bytes?} – external HTTP GET (captures stdout)\n"
    "  llm.prompt {prompt, model?, max_tokens?, endpoint?} – query local LLM (captures stdout)\n"
    "  browser.navigate {url} – open/navigate browser to URL\n"
    "  browser.dump {} – dump accessibility elements of active browser tab\n"
    "  browser.click {id} – click browser element by ID\n"
    "  browser.type {id, text} – type text into browser element\n"
    "  browser.screenshot {path?} – save browser screenshot to PNG"
)


def build_system_prompt(tools: list[str]) -> str:
    return (
        "You are the AI-Agent OS planner. Return ONE JSON object only. "
        "No markdown fences. No prose before or after the JSON.\n"
        "IMPORTANT: Your output is grammar-constrained. Emit ONLY the bare JSON object — no markdown, no prose.\n"
        f"Available tools: {tools}\n"
        "Schema:\n"
        '{"goal":string,"assumptions":[string],"success_criteria":[{"description":string,"tool":string,"args":{}}],'
        '"steps":[{"id":string,"purpose":string,"tool":string,"args":{},'
        '"depends_on":[string],"risk":"read_only"|"change",'
        '"captures":{varname:rule},"verify":[{"tool":string,"args":{}}]}]}\n'
        "Rules:\n"
        "- Every step with risk=change MUST have at least one verify entry.\n"
        "- Step ids: lowercase a-z, 0-9, hyphens only.\n"
        "- depends_on: list of step ids that must complete before this step runs.\n"
        "- captures: map from variable name to capture rule. Rules: stdout:strip, "
        "  stdout:first_line, exit_code, regex:<pattern>, json:<dot.path>\n"
        "- Use ${scratchpad.varname} in args to reference captured values from prior steps.\n"
        "- Use command.argv for tasks not covered by typed tools; "
        '  args must be {"argv":["binary","arg1",...]} where binary is in the safe allowlist.\n'
        f"{TOOL_TIPS}"
    )


def format_dataset(input_path: str, output_path: str, default_inventory: list | None = None):
    print(f"Formatting {input_path} -> {output_path} (Harness v2.3 ChatML format)...")
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file {input_path} does not exist.")
        return

    inv_text = json.dumps(default_inventory or [], ensure_ascii=False)
    chatml_entries = []
    skipped = 0

    counts = {"plan": 0, "chunk_phases": 0, "repair": 0}

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            item_type = item.get("type", "plan")

            # ── Mode A: Chunked Phase Decompositions ─────────────────────────
            if item_type == "chunk_phases":
                intent = item.get("intent")
                phases = item.get("phases")
                if not intent or not phases:
                    skipped += 1
                    continue

                sys_prompt = (
                    "You are the AI-Agent OS task decomposer. "
                    "Decompose the user intent into 2-5 ordered phases. "
                    "Return ONE JSON object only: "
                    '{"phases":[{"id":"phase1","goal":"...","tools_hint":["tool1",...]},...]}.\n'
                    "No prose, no markdown. Phases must be sequential and each produce a clear output."
                )
                user_prompt = f"Intent: {intent}\nSystem inventory: {inv_text}"
                assistant_content = json.dumps(phases, indent=2, ensure_ascii=False)

                chatml_entries.append({
                    "id": item.get("id", line_num),
                    "type": "chunk_phases",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_content}
                    ]
                })
                counts["chunk_phases"] += 1

            # ── Mode B: Step Error Recovery & Repair Traces ───────────────────
            elif item_type == "repair":
                ctx = item.get("context", {})
                plan_obj = item.get("plan")
                purpose = ctx.get("purpose", "Step")
                failure = ctx.get("failure", "Execution failed")
                evidence = ctx.get("evidence", [])

                if not plan_obj:
                    skipped += 1
                    continue

                ev_text = json.dumps(evidence[-4:], ensure_ascii=False)
                sys_prompt = (
                    f"Step '{purpose}' failed with: {failure}\n"
                    f"Evidence: {ev_text}\n"
                    "Return a JSON repair plan (same schema as a normal plan). "
                    f"Available tools: {sorted(ALLOWED_TOOLS)}. Max 3 steps. "
                    "Each system-changing step MUST have a verify entry. No prose, just JSON."
                )
                user_prompt = "Repair plan:"
                assistant_content = json.dumps(plan_obj, indent=2, ensure_ascii=False)

                chatml_entries.append({
                    "id": item.get("id", line_num),
                    "type": "repair",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_content}
                    ]
                })
                counts["repair"] += 1

            # ── Mode C: Standard & Multi-Step Execution Plans ────────────────
            else:
                intent = item.get("intent")
                plan_obj = item.get("plan")

                if not intent or not plan_obj:
                    skipped += 1
                    continue

                try:
                    validate_plan(plan_obj)
                except Exception:
                    skipped += 1
                    continue

                # In 80% of samples use category subsetting; in 20% use full catalogue
                # This teaches model to adapt to both filtered and full environments
                if random.random() < 0.8:
                    tools = _build_tool_subset(intent)
                else:
                    tools = sorted(ALLOWED_TOOLS)

                sys_prompt = build_system_prompt(tools)
                user_prompt = f"Intent: {intent}\nSystem inventory: {inv_text}"
                assistant_content = json.dumps(plan_obj, indent=2, ensure_ascii=False)

                chatml_entries.append({
                    "id": item.get("id", line_num),
                    "type": "plan",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_content}
                    ]
                })
                counts["plan"] += 1

    print(f"Validated and formatted {len(chatml_entries)} ChatML examples (skipped {skipped} invalid):")
    print(f"  - Plans: {counts['plan']}")
    print(f"  - Chunked phases: {counts['chunk_phases']}")
    print(f"  - Repair traces: {counts['repair']}")

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in chatml_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved ChatML training dataset to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format OS planning dataset into Harness v2.3 ChatML JSONL")
    parser.add_argument("--input", default="planning_dataset.jsonl", help="Input planning pairs JSONL")
    parser.add_argument("--output", default="planning_chatml.jsonl", help="Output ChatML JSONL")
    args = parser.parse_args()

    format_dataset(args.input, args.output)

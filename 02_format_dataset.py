import json
import argparse
import random

CANONICAL_SYSTEM_PROMPT = """You are an expert Linux kernel security AI intercepting syscall queries.
You must analyze the process query and return your verdict strictly as a single JSON object.
Do not include markdown fences, reasoning preambles, or conversational commentary.

Output Format Requirements:
1. "analysis": Exactly 1-2 concise sentences (strictly maximum 35 words) explaining the causal security policy.
2. "action": Strictly either "ALLOW" or "DENY".
3. "reason": A short category tag (maximum 6 words)."""

def format_chatml(input_file: str, output_file: str, balance: bool = True):
    print(f"Loading distilled data from {input_file}...")
    allow_samples = []
    deny_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            completion_raw = data.get("teacher_completion", "")
            if not completion_raw:
                continue

            try:
                parsed = json.loads(completion_raw)
            except json.JSONDecodeError:
                continue

            action = parsed.get("action", "").upper().strip()
            if action not in ("ALLOW", "DENY"):
                continue

            analysis = parsed.get("analysis", "").strip()
            if not analysis:
                continue

            # Ensure word count is strictly bounded
            words = analysis.split()
            if len(words) > 35:
                analysis = " ".join(words[:35]) + "."
                parsed["analysis"] = analysis

            compact_completion = json.dumps(parsed, ensure_ascii=False)

            messages = [
                {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
                {"role": "user", "content": data["prompt"]},
                {"role": "assistant", "content": compact_completion}
            ]
            item = {"messages": messages, "query_id": data.get("query_id")}

            if action == "ALLOW":
                allow_samples.append(item)
            else:
                deny_samples.append(item)

    print(f"Parsed valid samples: {len(allow_samples)} ALLOW, {len(deny_samples)} DENY.")

    if balance and allow_samples and deny_samples:
        min_count = min(len(allow_samples), len(deny_samples))
        random.seed(42)
        random.shuffle(allow_samples)
        random.shuffle(deny_samples)
        final_data = allow_samples[:min_count] + deny_samples[:min_count]
        random.shuffle(final_data)
        print(f"Balanced dataset: {min_count} ALLOW + {min_count} DENY = {len(final_data)} total examples.")
    else:
        final_data = allow_samples + deny_samples
        random.shuffle(final_data)

    print(f"Writing {len(final_data)} formatted examples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Formatting complete! Data is ready for supervised fine-tuning.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="distilled_raw.jsonl", help="Raw distilled JSONL")
    parser.add_argument("--output", default="distilled_chatml.jsonl", help="ChatML formatted JSONL for training")
    parser.add_argument("--no-balance", action="store_true", help="Do not balance ALLOW/DENY classes")
    args = parser.parse_args()
    
    format_chatml(args.input, args.output, balance=not args.no_balance)

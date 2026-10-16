import os
import json
import asyncio
from groq import AsyncGroq
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Limit concurrent requests to avoid rate limits
CONCURRENCY_LIMIT = 2
DEFAULT_MODEL = os.environ.get("GROQ_TEACHER_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are an expert Linux kernel security AI intercepting syscall queries.
You must analyze the process query and return your verdict strictly as a single JSON object.
Do not include markdown fences, reasoning preambles, or conversational commentary.

Output Format Requirements:
1. "analysis": Exactly 1-2 concise sentences (strictly maximum 35 words) explaining the causal security policy.
2. "action": Strictly either "ALLOW" or "DENY".
3. "reason": A short category tag (maximum 6 words).

Example for malicious access:
{"analysis": "Process 'test_syscall' (PID 1024) attempted to read '/etc/shadow'. Unprivileged access to user authentication hashes violates credential protection.", "action": "DENY", "reason": "Unauthorized access to /etc/shadow"}

Example for benign access:
{"analysis": "Process 'gcc' is reading standard library headers in '/usr/include'. This is normal compilation behavior.", "action": "ALLOW", "reason": "Normal compiler file read"}
"""

def validate_and_clean_teacher_response(raw_text: str) -> dict | None:
    if not raw_text:
        return None
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    action = data.get("action", "").upper().strip()
    if action not in ("ALLOW", "DENY"):
        return None

    analysis = data.get("analysis", "").strip()
    if not analysis:
        # Fallback to reason if analysis is missing
        analysis = data.get("reason", "Standard security evaluation.")

    # Guard: word count must not exceed 40 words (enforces bounded Micro-CoT)
    words = analysis.split()
    if len(words) > 40:
        # Truncate to first 35 words to guarantee the 135M model never sees bloated CoT
        analysis = " ".join(words[:35]) + "."

    reason = data.get("reason", "").strip() or ("Allowed" if action == "ALLOW" else "Security violation")

    return {
        "analysis": analysis,
        "action": action,
        "reason": reason
    }

async def fetch_teacher_response(client: AsyncGroq, sem: asyncio.Semaphore, query_id: int, prompt: str, model: str):
    async with sem:
        print(f"Requesting distillation for query_id: {query_id}")
        for attempt in range(10):
            try:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # GREEDY decoding: absolute lowest entropy, no creative rambling
                    max_tokens=256
                )
                raw_content = completion.choices[0].message.content
                clean_data = validate_and_clean_teacher_response(raw_content)
                if clean_data:
                    # Space out requests slightly
                    await asyncio.sleep(1.5)
                    return query_id, prompt, json.dumps(clean_data)
                else:
                    print(f"  [WARN] query_id {query_id}: Teacher returned invalid format. Retrying... (Attempt {attempt+1})")
                    await asyncio.sleep(2)
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "429" in err_str:
                    wait_time = 25 + (attempt * 10)
                    print(f"Rate limit for query_id {query_id}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"[ERROR] query_id {query_id}: {e}")
                    await asyncio.sleep(3)
        print(f"[ERROR] query_id {query_id}: Max retries reached.")
        return query_id, prompt, None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset.jsonl", help="Raw dataset file")
    parser.add_argument("--output", default="distilled_raw.jsonl", help="Output file with teacher responses")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Teacher model to query on Groq")
    parser.add_argument("--dry-run", action="store_true", help="Only run a few samples")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("[ERROR] GROQ_API_KEY not set. Copy .env.template to .env and set your key.")
        return

    client = AsyncGroq(api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    existing_queries = set()
    if os.path.exists(args.output):
        print(f"Loading existing {args.output} to resume...")
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    existing_queries.add(data.get("query_id"))
                except json.JSONDecodeError:
                    continue

    print(f"Loading {args.input}...")
    unique_queries = {}
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file '{args.input}' not found. Run 00_generate_synthetic_dataset.py first.")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            q_id = data.get("query_id")
            prompt = data.get("prompt")
            if prompt and q_id not in unique_queries and q_id not in existing_queries:
                unique_queries[q_id] = prompt

    items = list(unique_queries.items())
    if args.dry_run:
        items = items[:3]
        print("Dry run mode: processing only 3 items.")

    print(f"Found {len(items)} unique queries to distill using teacher {args.model} (temp=0.0).")
    if not items:
        print("No new queries to distill.")
        return

    tasks = [fetch_teacher_response(client, sem, q_id, p, args.model) for q_id, p in items]

    print(f"Saving distilled responses to {args.output}...")
    success_count = 0
    with open(args.output, "a", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            q_id, prompt, response = await coro
            if response:
                out = {
                    "query_id": q_id,
                    "prompt": prompt,
                    "teacher_completion": response
                }
                f.write(json.dumps(out) + "\n")
                f.flush()
                success_count += 1

    print(f"Successfully distilled {success_count}/{len(items)} queries.")
    print("Next step: Run 'python 02_format_dataset.py' to generate the balanced ChatML dataset.")

if __name__ == "__main__":
    asyncio.run(main())

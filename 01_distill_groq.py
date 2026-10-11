import os
import json
import asyncio
from groq import AsyncGroq
import argparse

# Limit concurrent requests to avoid rate limits
CONCURRENCY_LIMIT = 2

SYSTEM_PROMPT = """You are an expert Linux kernel security AI.
You must analyze the following syscall/process event query and provide a safety verdict.
First, provide a brief Chain-of-Thought (CoT) reasoning block analyzing the risk.
Then, output your final verdict in exact JSON format.
Example output format:
Reasoning: The process is test_syscall and it is opening /etc/shadow which is a highly sensitive authentication file. This is dangerous unless it's a known authentication daemon.
Verdict: {"action": "DENY", "reason": "Unauthorized access to /etc/shadow"}
"""

async def fetch_teacher_response(client: AsyncGroq, sem: asyncio.Semaphore, query_id: int, prompt: str):
    async with sem:
        print(f"Requesting distillation for query_id: {query_id}")
        for attempt in range(10):
            try:
                completion = await client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=512
                )
                # Sleep a bit after a successful request to space them out
                await asyncio.sleep(2)
                return query_id, prompt, completion.choices[0].message.content
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "429" in err_str:
                    wait_time = 30 + (attempt * 15)
                    print(f"Rate limit for query_id {query_id}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"[ERROR] query_id {query_id}: {e}")
                    return query_id, prompt, None
        print(f"[ERROR] query_id {query_id}: Max retries reached.")
        return query_id, prompt, None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset.jsonl", help="Raw dataset file")
    parser.add_argument("--output", default="distilled_raw.jsonl", help="Output file with teacher responses")
    parser.add_argument("--dry-run", action="store_true", help="Only run a few samples")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY not set. Export it before running this script.")
        return

    client = AsyncGroq(api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    existing_queries = set()
    if os.path.exists(args.output):
        print(f"Loading existing {args.output} to resume...")
        with open(args.output, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    existing_queries.add(data.get("query_id"))
                except json.JSONDecodeError:
                    continue

    # Load raw unique prompts
    print(f"Loading {args.input}...")
    unique_queries = {}
    with open(args.input, "r") as f:
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

    print(f"Found {len(items)} unique queries to distill.")

    tasks = [fetch_teacher_response(client, sem, q_id, p) for q_id, p in items]

    # Save to output incrementally
    print(f"Saving distilled responses to {args.output}...")
    success_count = 0
    with open(args.output, "a") as f:
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

if __name__ == "__main__":
    asyncio.run(main())

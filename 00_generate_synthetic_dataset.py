import os
import json
import asyncio
import time
import random
import argparse
from datetime import datetime, timezone
from groq import AsyncGroq

# Limit concurrent requests
CONCURRENCY_LIMIT = 3
DEFAULT_BATCHES = 10  # 10 batches of 20 = 200 synthetic prompts

SYSTEM_PROMPT = """You are an expert Linux kernel developer generating synthetic telemetry datasets.
Generate exactly 20 diverse, realistic process queries that a kernel-level AI agent might intercept via sys_agent_query.
Provide exactly 20 JSON objects in a JSON array. Each object must have two fields:
"comm": the process name (e.g., "nginx", "bash", "nc", "test_syscall", "gcc", "sshd")
"query": the detailed query describing what the process is trying to do.

Vary the scenarios:
- 40% Benign system activities (compiling, web servers, DNS, writing to /tmp)
- 40% Malicious activities (reverse shells, unauthorized access to /etc/shadow, injecting code)
- 20% Edge cases (stress testing threads, resource exhaustion, weird kernel behavior)

Output ONLY a valid JSON array of objects. Do not include markdown formatting (like ```json), just the raw array.
"""

def generate_full_prompt(comm, query):
    pid = random.randint(1000, 9999)
    return f"""You are the operating system's kernel AI agent. A process is querying you via sys_agent_query:
- Process Comm: {comm}
- Caller PID: {pid}
- Target PID: {pid}
- Query: {query}

Provide a concise, direct, 2-3 sentence system safety analysis and actionable recommendation."""

async def generate_batch(client: AsyncGroq, sem: asyncio.Semaphore, batch_id: int):
    async with sem:
        print(f"Requesting synthetic batch {batch_id}...")
        try:
            completion = await client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Generate a new diverse batch of 20 queries."}
                ],
                temperature=0.7, # Higher temperature for variety
                max_tokens=4000
            )
            content = completion.choices[0].message.content.strip()
            # Strip markdown if model accidentally adds it
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            data = json.loads(content)
            return data
        except Exception as e:
            print(f"[ERROR] Batch {batch_id} failed: {e}")
            return []

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset.jsonl", help="Dataset file to append to")
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES, help="Number of batches (20 items each) to generate")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY not set. Export it before running this script.")
        return

    client = AsyncGroq(api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    print(f"Starting generation of {args.batches} batches (approx {args.batches * 20} items)...")
    tasks = [generate_batch(client, sem, i+1) for i in range(args.batches)]
    results = await asyncio.gather(*tasks)

    generated_items = []
    for batch_data in results:
        if isinstance(batch_data, list):
            generated_items.extend(batch_data)

    print(f"\nSuccessfully generated {len(generated_items)} raw scenarios.")
    
    # Get max query_id from existing dataset
    start_id = 100
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    q_id = data.get("query_id", 0)
                    if q_id > start_id:
                        start_id = q_id
                except:
                    pass
    start_id += 1

    print(f"Appending to {args.output} starting at query_id {start_id}...")
    
    success_count = 0
    with open(args.output, "a") as f:
        for item in generated_items:
            comm = item.get("comm", "unknown")
            query = item.get("query")
            if not query:
                continue
                
            full_prompt = generate_full_prompt(comm, query)
            out_obj = {
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "prompt": full_prompt,
                "completion": None,
                "model": "smollm2-135m-instruct",
                "query_id": start_id + success_count,
                "comm": comm,
                "latency_ms": round(random.uniform(5.0, 50.0), 2),
                "outcome": None
            }
            f.write(json.dumps(out_obj) + "\n")
            success_count += 1

    print(f"Done! Appended {success_count} new queries to {args.output}.")
    print("Next step: Run 'python 01_distill_groq.py' to distill the teacher responses for these new queries.")

if __name__ == "__main__":
    asyncio.run(main())

import os
import json
import asyncio
import time
import random
import argparse
from datetime import datetime, timezone
from groq import AsyncGroq

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Limit concurrent requests to avoid rate limits
CONCURRENCY_LIMIT = 2
DEFAULT_BATCHES = 10  # 10 batches of 20 = 200 synthetic prompts
DEFAULT_MODEL = os.environ.get("GROQ_TEACHER_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are an expert Linux kernel developer generating synthetic telemetry datasets.
Generate exactly 20 diverse, realistic process queries that a kernel-level AI agent might intercept via sys_agent_query.
Provide exactly 20 JSON objects in a JSON array. Each object must have two fields:
"comm": the process name (e.g., "nginx", "bash", "nc", "test_syscall", "gcc", "sshd", "python3", "curl", "systemd-resolved")
"query": the detailed query describing what the process is trying to do.

Vary the scenarios strictly:
- 45% Benign system activities (compiling C code, binding port 80/443, local DNS query, reading shared libraries in /usr/lib, user reading ~/Documents)
- 45% Malicious / Suspicious activities (reverse shell to external IP, unprivileged reading /etc/shadow, writing to /dev/mem, injecting ptrace into root process, overwriting /boot/extlinux.conf)
- 10% Ambiguous edge cases (monitoring utilities reading /proc, cron runner restarting service, stress test spawned processes)

Output ONLY a valid JSON array of objects. Do not include markdown formatting (like ```json), just the raw array.
"""

def generate_full_prompt(comm, query):
    pid = random.randint(1000, 9999)
    return f"""You are the operating system's kernel AI agent. A process is querying you via sys_agent_query:
- Process Comm: {comm}
- Caller PID: {pid}
- Target PID: {pid}
- Query: {query}

Provide a concise, direct system safety analysis and verdict."""

async def generate_batch(client: AsyncGroq, sem: asyncio.Semaphore, batch_id: int, model: str):
    async with sem:
        print(f"Requesting synthetic batch {batch_id} using {model}...")
        for attempt in range(10):
            try:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "Generate a new diverse batch of 20 queries."}
                    ],
                    temperature=0.2,
                    max_tokens=3000
                )
                content = completion.choices[0].message.content.strip()
                if content.startswith("```json"): content = content[7:]
                if content.startswith("```"): content = content[3:]
                if content.endswith("```"): content = content[:-3]
                content = content.strip()
                
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                print(f"[WARN] Batch {batch_id} returned non-list JSON.")
                return []
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate limit" in err_str.lower():
                    wait_time = 20 + attempt * 10
                    print(f"Rate limit on batch {batch_id}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"[ERROR] Batch {batch_id} attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(3)
        print(f"[ERROR] Batch {batch_id} max retries reached.")
        return []

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset.jsonl", help="Dataset file to append to")
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES, help="Number of batches (20 items each) to generate")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model to use for generation")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("[ERROR] GROQ_API_KEY not set. Copy .env.template to .env and set your key.")
        return

    client = AsyncGroq(api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    print(f"Starting generation of {args.batches} batches (approx {args.batches * 20} items) with model {args.model}...")
    tasks = [generate_batch(client, sem, i+1, args.model) for i in range(args.batches)]
    results = await asyncio.gather(*tasks)

    generated_items = []
    for batch_data in results:
        if isinstance(batch_data, list):
            generated_items.extend(batch_data)

    print(f"\nSuccessfully generated {len(generated_items)} raw scenarios.")
    
    start_id = 100
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
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
    with open(args.output, "a", encoding="utf-8") as f:
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
    print("Next step: Run 'python 01_distill_groq.py' to distill the teacher responses.")

if __name__ == "__main__":
    asyncio.run(main())

# Distillation + Fine Tuning Pipeline

> The OS locally runs a SmolLM2 135M model for the OS AI Agent. We fine-tune it for the OS AI agent to make it better at the low-level specific job using Knowledge Distillation.

This repository contains the end-to-end pipeline for **Sequence-Level Knowledge Distillation**. Because the local `SmolLM2-135M-Instruct` model is heavily constrained, we use a much larger teacher model (`Llama 3.1 70B` via Groq) to label our raw OS telemetry and provide Chain-of-Thought (CoT) reasoning. We then fine-tune our local model to mimic this expert reasoning path.

---

## Prerequisites

1. Install Python requirements (ensure you have PyTorch, HuggingFace transformers, and PEFT installed):

   ```bash
   pip install groq datasets transformers peft trl torch
   ```

2. You must have a **Groq API Key** to run the teacher model distillation.

   ```powershell
   $env:GROQ_API_KEY="your-groq-api-key"
   ```

   *(On Linux/macOS: `export GROQ_API_KEY="your-groq-api-key"`)*

---

## Pipeline Execution

The distillation pipeline supports two complementary agent roles:

* **Track A — Kernel Security Auditor (`os_agent_lora.gguf`)**: Evaluates low-level syscalls (`sys_agent_query` Syscall 548) and issues `ALLOW` / `DENY` security verdicts.
* **Track B — Autonomous OS Task Planner (`os_planner_lora.gguf`)**: Decomposes natural language user intents into structured, policy-verified execution plans (`dir.create`, `file.write`, `package.install`, `service.start`, etc.).

---

### Track A: Syscall Security Pipeline

1. **Data Distillation (`01_distill_groq.py`)**:
   Queries Groq teacher model (`Llama 3.1 70B` or `Llama 3.3 70B`) on raw syscall events.

   ```bash
   python 01_distill_groq.py
   ```

2. **Format Dataset (`02_format_dataset.py`)**:
   Converts teacher responses into ChatML:

   ```bash
   python 02_format_dataset.py --input distilled_raw.jsonl --output distilled_chatml.jsonl
   ```

3. **Supervised Fine-Tuning (`train_lora.py`)**:

   ```bash
   python train_lora.py --dataset distilled_chatml.jsonl --output-dir ./adapters/os_agent_lora
   ```

4. **Evaluate Agent (`03_eval_distilled_agent.py`)**:

   ```bash
   python 03_eval_distilled_agent.py
   ```

---

### Track B: Autonomous Task Planner Pipeline (Harness v2.3 Aligned)

1. **Generate Synthetic Planning Dataset (`generate_planning_dataset.py`)**:
   Generates rich OS task scenarios across all 9 intent categories (files, document, data, browser, email, coding, service, system, research), including multi-step chains with scratchpad captures, chunked phase decomposition, and error repairs:

   ```bash
   python generate_planning_dataset.py --count 1500 --output planning_dataset.jsonl
   ```

2. **Format into ChatML (`format_planning_dataset.py`)**:
   Formats planning scenarios into `harness_v2.plan_prompt` ChatML format with dynamic category-filtered tool catalogues matching the inference-time harness:

   ```bash
   python format_planning_dataset.py --input planning_dataset.jsonl --output planning_chatml.jsonl
   ```

3. **Train Planning LoRA (`train_lora.py`)**:
   Fine-tunes the model on task planning with expanded sequence length (up to 1536 tokens) to prevent plan truncation:

   ```bash
   python train_lora.py --dataset planning_chatml.jsonl --output-dir ./adapters/os_planner_lora --epochs 3 --max-length 1536 --val-split 0.05
   ```

4. **Evaluate Planning Agent (`eval_planning_agent.py`)**:
   Benchmarks JSON parse rate, strict schema validity, tool catalogue compliance, and verification coverage:

   ```bash
   python eval_planning_agent.py --adapter ./adapters/os_planner_lora --dataset planning_dataset.jsonl --samples 25
   ```

5. **Export to GGUF (`export_gguf.py`)**:
   Converts PEFT adapter to GGUF format for in-guest `llama-server` (constrained by `plan_grammar.gbnf`):

   ```bash
   python export_gguf.py --lora-dir adapters/os_planner_lora --output-file os_planner_lora.gguf --quantization Q4_K_M
   ```

**Run the Updated fine tuning pipeline**

```bash
# Navigate to the distillation directory
cd c:\qemu-alpine\distillation-fine-tuning

# Step 1: Generate 1,500 validated task scenarios (plans + chunk phases + repair traces)
python generate_planning_dataset.py --count 1500 --output planning_dataset.jsonl

# Step 2: Format into Harness v2.3 ChatML instruction dataset
python format_planning_dataset.py --input planning_dataset.jsonl --output planning_chatml.jsonl

# Step 3: Fine-tune SmolLM2 with LoRA (expanded context length)
python train_lora.py --dataset planning_chatml.jsonl --output-dir ./adapters/os_planner_lora --epochs 3 --max-length 1536 --val-split 0.05

# Step 4: Benchmark the trained adapter
python eval_planning_agent.py --adapter ./adapters/os_planner_lora --dataset planning_dataset.jsonl --samples 25

# Step 5: Export to quantized GGUF for in-guest llama-server
python export_gguf.py --lora-dir adapters/os_planner_lora --output-file os_planner_lora.gguf --quantization Q4_K_M
```

# Distillation and Fine-Tuning Pipeline

> Autonomous Task Planning (Harness v2.3) and Kernel Syscall Security Adaptation for AI-Agent OS.

This repository contains the end-to-end dataset generation, knowledge distillation, LoRA fine-tuning, evaluation, and GGUF quantization pipeline for AI-Agent OS. The target model is `HuggingFaceTB/SmolLM2-135M-Instruct`, specialized for local in-guest inference inside Alpine Linux via `llama-server`.

---

## Architecture Overview

The fine-tuning pipeline supports two complementary operating system roles:

```
+-----------------------------------------------------------------------------------+
|                                 AI-Agent OS Roles                                 |
+-----------------------------------------+-----------------------------------------+
| Track A: Kernel Syscall Security Auditor| Track B: Autonomous Task Planner        |
| - Model: os_agent_lora.gguf (39 MB)     | - Model: os_planner_lora.gguf (100.6 MB) |
| - Intercepts: Syscall #548              | - Intercepts: Natural language goals    |
| - Action: ALLOW / DENY security verdicts| - Action: Policy-verified DAG JSON plans|
| - Teacher: Llama-3.1-70B via Groq       | - Alignment: Harness v2.3 Engine        |
+-----------------------------------------+-----------------------------------------+
```

1. **Track A: Kernel Security Auditor (`os_agent_lora.gguf`)**
   Evaluates low-level process queries and syscall invocations (`sys_agent_query` Syscall #548), issuing immediate `ALLOW` or `DENY` security verdicts with concise risk rationales.
2. **Track B: Autonomous Task Planner (`os_planner_lora.gguf`)**
   Decomposes complex human intents into deterministic, policy-verified DAG execution plans across 38 allowed system tools in 9 categories, strictly conforming to the Harness v2.3 specification.

---

## Prerequisites and Environment Setup

### 1. Python Environment

Ensure Python 3.10+ is installed with PyTorch and Hugging Face libraries:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft trl datasets accelerate groq
```

[NOTE] For GPU training, ensure CUDA-enabled PyTorch is installed (e.g. `torch-2.6.0+cu124`). On an NVIDIA RTX GPU with FP16 tensor cores, step times drop from ~3 hours (CPU) to ~6.8 seconds (GPU) - a 1,500x speedup.

### 2. Local GGUF Quantization Tool

The GGUF exporter automatically detects `llama-quantize` from your system path or local installations (such as Ollama or llama.cpp builds):
- Path: `llama-quantize` or `llama.cpp/llama-quantize` or `%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-quantize.exe`.

### 3. API Keys (Track A Only)

Track A utilizes Groq for teacher distillation:

```powershell
$env:GROQ_API_KEY="your-groq-api-key"
```

*(On Linux/macOS: `export GROQ_API_KEY="your-groq-api-key"`)*

---

## Track B: Autonomous Task Planner Pipeline (Harness v2.3 Aligned)

Track B aligns `SmolLM2-135M-Instruct` with the Harness v2.3 task execution engine, training it to produce schema-valid JSON plans without syntax failure or missing verification blocks.

### Step 1: Synthetic Planning Dataset Generation (`generate_planning_dataset.py`)

Generates 1,500 synthetically verified task planning scenarios across all 9 intent categories:
- **Categories Covered**: `files`, `document`, `data`, `browser`, `email`, `coding`, `service`, `system`, `research`.
- **Tools Exercised**: All 38 allowed system tools defined in `PolicyEngine.READ_ONLY` and `PolicyEngine.CHANGE`.
- **Plan Types**:
  - Standard multi-step execution plans with step dependencies (`depends_on`).
  - Chunk phase decompositions for complex multi-part goals (`chunk_phases`).
  - Self-repair traces correcting previous step errors (`repair_prompt`).
- **Policy Enforcement**:
  - Strict step risks: `risk: "read_only" | "change"`.
  - Mandatory verification blocks on all change actions (`verify: {"tool": ..., "args": ...}`).
  - Scratchpad variable chaining (`${scratchpad.<variable>}`).
- **Validation**: 100% of samples are checked with `harness_v2.validate_plan()` prior to export (0 syntax or schema errors).

Command:
```bash
python generate_planning_dataset.py --count 1500 --output planning_dataset.jsonl
```

### Step 2: Dynamic Category-Filtered ChatML Formatting (`format_planning_dataset.py`)

Converts the dataset into ChatML instruction turns (`system`, `user`, `assistant`).

Crucially, this formatter imports `harness_v2.classify_intent_categories()` and `harness_v2._build_tool_subset()` at generation time. By filtering the system prompt's tool catalog to the 1-3 categories relevant to the user's intent, prompt length is reduced by ~70%:
- **Median Token Count**: 1,066 tokens.
- **95th Percentile**: 1,650 tokens.
- **Maximum Token Count**: 1,786 tokens (guaranteed within the 2,048 sequence limit).

Command:
```bash
python format_planning_dataset.py --input planning_dataset.jsonl --output planning_chatml.jsonl
```

### Step 3: Hardware-Accelerated LoRA Fine-Tuning (`train_lora.py`)

Trains LoRA adapters on top of `HuggingFaceTB/SmolLM2-135M-Instruct` using Hugging Face `TRL` (`SFTTrainer`):
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Rank / Alpha**: `r=16`, `lora_alpha=32`, `dropout=0.05`.
- **Sequence Length**: `max_length=2048` (ensures multi-step plans and tool signatures are not truncated).
- **Precision**: `fp16=True` when CUDA is available.
- **Validation**: 5% evaluation split with token accuracy evaluation at every epoch.
- **Performance Benchmark**:
  - 270 steps (3 epochs, batch size 16 with gradient accumulation 4).
  - Training duration: 27.2 minutes on NVIDIA RTX 4050 GPU (~6.8s per step).
  - Final training loss: `0.2605`.
  - Eval loss: `0.0200`.
  - Eval token accuracy: `99.61%`.

Command:
```bash
python train_lora.py --dataset planning_chatml.jsonl --output-dir ./adapters/os_planner_lora --epochs 3 --max-length 2048 --val-split 0.05
```

### Step 4: Quantitative Evaluation Benchmark (`eval_planning_agent.py`)

Evaluates the fine-tuned adapter against held-out task scenarios:
- Verifies JSON parse rates and structural schema validity.
- Measures tool catalog adherence and verification coverage on state-modifying actions.
- Tests resilience across multi-step chains and edge cases.

Command:
```bash
python eval_planning_agent.py --adapter ./adapters/os_planner_lora --dataset planning_dataset.jsonl --samples 25
```

[NOTE] At runtime in the OS guest, `llama-server` is additionally constrained by `plan_grammar.gbnf`. This deterministic grammar eliminates invalid JSON syntax or unclosed braces on edge-case generations.

### Step 5: GGUF Model Export and Quantization (`export_gguf.py`)

Converts the trained PEFT LoRA adapter into an optimized, self-contained GGUF model:
1. Loads base `SmolLM2-135M-Instruct` and merges LoRA weights in FP16.
2. Converts the merged model to GGUF using `llama.cpp` conversion scripts.
3. Quantizes weights to `Q4_K_M` via `llama-quantize`.
4. Outputs: `os_planner_lora.gguf` (100.57 MB).

Command:
```bash
python export_gguf.py --lora-dir adapters/os_planner_lora --output-file os_planner_lora.gguf --quantization Q4_K_M
```

---

## Track A: Syscall Security Auditor Pipeline

For evaluating kernel-level syscall queries (Syscall #548):

1. **Distill from Teacher Model (`01_distill_groq.py`)**:
   ```bash
   python 01_distill_groq.py
   ```
2. **Format Dataset (`02_format_dataset.py`)**:
   ```bash
   python 02_format_dataset.py --input distilled_raw.jsonl --output distilled_chatml.jsonl
   ```
3. **Train Security LoRA (`train_lora.py`)**:
   ```bash
   python train_lora.py --dataset distilled_chatml.jsonl --output-dir ./adapters/os_agent_lora --epochs 3 --max-length 512
   ```
4. **Evaluate Agent (`03_eval_distilled_agent.py`)**:
   ```bash
   python 03_eval_distilled_agent.py
   ```

---

## One-Command End-to-End Execution

To regenerate the dataset, format it, fine-tune the model, and export the GGUF model:

```bash
# 1. Generate 1,500 validated task scenarios
python generate_planning_dataset.py --count 1500 --output planning_dataset.jsonl

# 2. Format with dynamic category tool subsets
python format_planning_dataset.py --input planning_dataset.jsonl --output planning_chatml.jsonl

# 3. Fine-tune on GPU with 2048 sequence length
python train_lora.py --dataset planning_chatml.jsonl --output-dir ./adapters/os_planner_lora --epochs 3 --max-length 2048 --val-split 0.05

# 4. Run evaluation benchmark
python eval_planning_agent.py --adapter ./adapters/os_planner_lora --dataset planning_dataset.jsonl --samples 25

# 5. Export and quantize to GGUF
python export_gguf.py --lora-dir adapters/os_planner_lora --output-file os_planner_lora.gguf --quantization Q4_K_M
```

---

## Deployment to Alpine VM

Once `os_planner_lora.gguf` is created:

1. **Direct SCP Upload**:
   ```bash
   scp -P 2222 os_planner_lora.gguf root@127.0.0.1:/var/lib/ai-agent/models/
   ```

2. **Configure OpenRC Service**:
   Inside the Alpine VM, edit `/etc/init.d/llama-server` or set `/etc/conf.d/llama-server`:
   ```bash
   MODEL_PATH="/var/lib/ai-agent/models/os_planner_lora.gguf"
   ```

3. **Restart Inference Server**:
   ```bash
   rc-service llama-server restart
   ```

---

## Repository File Map

| File | Type | Description |
|---|---|---|
| `generate_planning_dataset.py` | Generator | Synthesizes 1,500 validated multi-step plans, chunk phases, and repair traces adhering to Harness v2.3. |
| `format_planning_dataset.py` | Preprocessor | Formats planning dataset into ChatML with dynamic intent-category tool filtering. |
| `train_lora.py` | Trainer | Hardware-accelerated PEFT LoRA trainer with FP16, 2048 sequence budget, and accuracy monitoring. |
| `eval_planning_agent.py` | Benchmark | Evaluates adapter JSON parse rate, schema validity, and tool constraints on held-out tasks. |
| `export_gguf.py` | Exporter | Merges LoRA weights, converts to GGUF, and quantizes to `Q4_K_M` via `llama-quantize`. |
| `00_generate_synthetic_dataset.py` | Generator | Generates synthetic kernel syscall events for Track A. |
| `01_distill_groq.py` | Distiller | Queries Groq teacher model (`Llama 3.1 70B`) for Track A syscall reasoning. |
| `02_format_dataset.py` | Preprocessor | Formats Track A distilled queries into ChatML format. |
| `03_eval_distilled_agent.py` | Benchmark | Evaluates Track A security auditor accuracy against ground-truth verdicts. |
| `planning_dataset.jsonl` | Dataset | 1,500 validated task scenarios (1.58 MB). |
| `planning_chatml.jsonl` | Dataset | Formatted ChatML dataset for Track B fine-tuning (6.05 MB). |
| `os_planner_lora.gguf` | Artifact | Fused, Q4_K_M quantized planner model for in-guest execution (100.57 MB). |
| `os_agent_lora.gguf` | Artifact | Quantized LoRA adapter for Track A syscall security auditor (39.1 MB). |

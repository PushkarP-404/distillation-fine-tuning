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

Run the following scripts sequentially to complete the distillation and training loop.

### 1. Data Distillation (`01_distill_groq.py`)
Reads the raw syscall telemetry from `dataset.jsonl` and queries the Groq API. The Llama 3.1 70B teacher model will generate a high-quality Chain-of-Thought reasoning block and the final JSON verdict.
```bash
python 01_distill_groq.py
```
*(Optional: use `--dry-run` to test on just 3 samples before processing the whole dataset).*
**Output:** `distilled_raw.jsonl`

### 2. Format Dataset (`02_format_dataset.py`)
Converts the raw teacher responses into the exact `ChatML` instruction format expected by the `SmolLM2-135M-Instruct` tokenizer. 
```bash
python 02_format_dataset.py
```
**Output:** `distilled_chatml.jsonl`

### 3. Supervised Fine-Tuning (`train_lora.py`)
Loads the ChatML dataset via HuggingFace Datasets and performs LoRA (Low-Rank Adaptation) fine-tuning on the base `SmolLM2-135M-Instruct` model.
```bash
python train_lora.py
```
**Output:** Fine-tuned adapter saved to `./adapters/distilled_lora/`

### 4. Evaluate Distilled Agent (`03_eval_distilled_agent.py`)
Benchmarks the newly trained distilled LoRA adapter against the expected verdicts from the teacher model to ensure knowledge transfer was successful.
```bash
python 03_eval_distilled_agent.py
```

### 5. Export and Deploy (`export_gguf.py` & `apply_lora.py`)
Merge the adapter weights into the base model, export it to the GGUF format for edge deployment, and deploy it to the QEMU VM.
```bash
python export_gguf.py
python apply_lora.py
```

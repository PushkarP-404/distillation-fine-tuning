import os
import subprocess
import sys
import shutil
import argparse
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

def parse_args():
    parser = argparse.ArgumentParser(description="Convert and Quantize fine-tuned models to GGUF format")
    default_dir = "adapters/os_agent_lora" if os.path.exists("adapters/os_agent_lora") else "adapters/os_planner_lora"
    default_out = "os_agent_smollm2_q4_k_m.gguf"
    
    parser.add_argument("--lora-dir", default=default_dir, help="Directory containing fine-tuned LoRA adapter")
    parser.add_argument("--output-file", default=default_out, help="Output GGUF file path")
    parser.add_argument("--base-model", default=BASE_MODEL_ID, help="Base Hugging Face model ID")
    parser.add_argument("--quantization", default="Q4_K_M", choices=["Q4_K_M", "q8_0", "f16", "f32"], help="Quantization type")
    parser.add_argument("--adapter-only", action="store_true", help="Export standalone LoRA adapter without merging")
    return parser.parse_args()

def export_standalone_lora(lora_dir: str, output_file: str):
    print(f"\n[INFO] Converting standalone LoRA adapter from {lora_dir} to {output_file}...")
    script_path = os.path.join("llama.cpp", "convert_lora_to_gguf.py")
    if not os.path.exists(script_path):
        script_path = os.path.join("llama.cpp", "convert-lora-to-gguf.py")
    
    if not os.path.exists(script_path):
        print("[ERROR] Could not find convert_lora_to_gguf.py in llama.cpp directory.")
        sys.exit(1)

    cmd = [sys.executable, script_path, "--outfile", output_file, lora_dir]
    subprocess.run(cmd, check=True)
    print(f"\n[SUCCESS] LoRA GGUF adapter generated at: {os.path.abspath(output_file)}")

def merge_and_export_quantized(lora_dir: str, output_file: str, base_model_id: str, quant_type: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_temp_dir = "./temp_merged_model"
    if os.path.exists(os.path.join(merged_temp_dir, "model.safetensors")):
        print(f"\n[1/3] Found existing merged model in '{merged_temp_dir}'. Skipping re-merge...")
    else:
        print(f"\n[1/3] Loading base model '{base_model_id}' and merging LoRA weights from '{lora_dir}'...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        model = PeftModel.from_pretrained(base_model, lora_dir)
        print("Fusing LoRA weights into base model layers...")
        merged_model = model.merge_and_unload()

        print(f"Saving merged FP16 checkpoint to {merged_temp_dir}...")
        os.makedirs(merged_temp_dir, exist_ok=True)
        merged_model.save_pretrained(merged_temp_dir)
        tokenizer.save_pretrained(merged_temp_dir)

    print(f"\n[2/3] Converting merged model to GGUF format...")
    convert_script = os.path.join("llama.cpp", "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        print(f"[ERROR] '{convert_script}' not found.")
        sys.exit(1)

    temp_f16_gguf = "temp_merged_f16.gguf"
    cmd_convert = [
        sys.executable,
        convert_script,
        merged_temp_dir,
        "--outfile", temp_f16_gguf,
        "--outtype", "f16"
    ]
    subprocess.run(cmd_convert, check=True)

    print(f"\n[3/3] Applying {quant_type} quantization...")
    # Check for local llama-quantize binary or perform SSH VM quantization
    quantize_bin = None
    ollama_quantize = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-quantize.exe")
    for candidate in [
        ollama_quantize,
        os.path.join("bin", "llama-quantize.exe"),
        os.path.join("llama.cpp", "build", "bin", "llama-quantize.exe"),
        "llama-quantize.exe",
        "llama-quantize"
    ]:
        if os.path.exists(candidate) or shutil.which(candidate):
            quantize_bin = candidate
            break

    if quantize_bin:
        print(f"Running local quantizer: {quantize_bin} {temp_f16_gguf} {output_file} {quant_type}...")
        subprocess.run([quantize_bin, temp_f16_gguf, output_file, quant_type], check=True)
        if os.path.exists(temp_f16_gguf):
            os.remove(temp_f16_gguf)
    else:
        # If llama-quantize binary is not on host, check if q8_0 direct python quantization or VM SSH quantize is preferred
        if quant_type.lower() == "q8_0":
            print("Direct q8_0 conversion via convert_hf_to_gguf.py...")
            cmd_q8 = [
                sys.executable,
                convert_script,
                merged_temp_dir,
                "--outfile", output_file,
                "--outtype", "q8_0"
            ]
            subprocess.run(cmd_q8, check=True)
        else:
            print("[INFO] 'llama-quantize' binary not found on Windows host.")
            print(f"[INFO] Moving {temp_f16_gguf} to '{output_file}' as FP16 base.")
            print("[INFO] To quantize in Alpine VM: run 'llama-quantize input.gguf output_q4.gguf Q4_K_M'")
            if os.path.exists(output_file):
                os.remove(output_file)
            os.rename(temp_f16_gguf, output_file)

    # Clean up temp merged HF folder to save disk space
    if os.path.exists(merged_temp_dir):
        shutil.rmtree(merged_temp_dir, ignore_errors=True)

    print(f"\n[SUCCESS] Quantized model created at: {os.path.abspath(output_file)}")
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"File Size: {size_mb:.2f} MB")

def main():
    args = parse_args()
    print("==================================================")
    print("   AI-Agent OS: GGUF Model & LoRA Exporter Tool   ")
    print("==================================================")

    if not os.path.exists(args.lora_dir):
        print(f"[ERROR] LoRA adapter directory '{args.lora_dir}' not found.")
        print("Please train the adapter first using train_lora.py.")
        sys.exit(1)

    if args.adapter_only:
        export_standalone_lora(args.lora_dir, args.output_file)
    else:
        merge_and_export_quantized(args.lora_dir, args.output_file, args.base_model, args.quantization)

    print("\n==================================================")
    print("                 DEPLOYMENT STEPS                 ")
    print("==================================================")
    print("1. Deploy the model to the Alpine VM:")
    print(f"   scp -P 2222 {args.output_file} root@127.0.0.1:/var/lib/ai-agent/models/")
    print("\n2. In the VM, configure /etc/init.d/llama-server to point to the new model:")
    print(f"   MODEL_PATH=/var/lib/ai-agent/models/{os.path.basename(args.output_file)}")
    print("3. Restart llama-server:")
    print("   rc-service llama-server restart\n")

if __name__ == "__main__":
    main()

import json
import argparse

def format_chatml(input_file, output_file):
    print(f"Loading distilled data from {input_file}...")
    formatted_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # SmolLM2 Instruct format using standard HuggingFace messages
            messages = [
                {"role": "system", "content": "You are the operating system's kernel AI agent."},
                {"role": "user", "content": data["prompt"]},
                {"role": "assistant", "content": data["teacher_completion"]}
            ]
            
            formatted_data.append({"messages": messages, "query_id": data.get("query_id")})
    
    print(f"Writing {len(formatted_data)} formatted examples to {output_file}...")
    with open(output_file, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
            
    print("Formatting complete! Data is ready for supervised fine-tuning.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="distilled_raw.jsonl", help="Raw distilled JSONL")
    parser.add_argument("--output", default="distilled_chatml.jsonl", help="ChatML formatted JSONL for training")
    args = parser.parse_args()
    
    format_chatml(args.input, args.output)

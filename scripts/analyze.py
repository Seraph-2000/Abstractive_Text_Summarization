import sys
import os
import torch
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from datasets import load_dataset

# --- PATH SETUP ---
# Ensure we can import from 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.utils import load_config
from src.verifier import HallucinationVerifier

def main():
    # 1. Load Config
    config_path = os.path.join(project_root, "config.yaml")
    config = load_config(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load the Model (Base + Adapter)
    adapter_path = os.path.join(project_root, config['output_dir'])
    
    print(f"Loading base model: {config['model_name']}...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config['model_name'])
    
    print(f"Loading LoRA adapter from: {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"\n❌ Error loading adapter: {e}")
        print(f"Did you run 'train.py' first? Checking folder: {adapter_path}")
        return

    model.to(device)
    model.eval()

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # 3. Initialize the Hallucination Verifier
    verifier = HallucinationVerifier(device=device)

    # 4. Load Dataset
    print(f"Loading {config['dataset_name']} dataset for inference...")
    dataset_split = "validation" 
    # XSum uses 'validation', CNN/DM uses 'validation' (mapped in 3.0.0)
    
    if config['dataset_name'] == "cnn_dailymail":
        ds = load_dataset(config['dataset_name'], config['dataset_config'], split=dataset_split)
    else:
        ds = load_dataset(config['dataset_name'], split=dataset_split)

    # 5. Run Analysis Loop
    num_samples_to_test = 5 
    print(f"Starting analysis on {num_samples_to_test} random samples...")

    for i in range(num_samples_to_test):
        print(f"\n\n>>> ANALYZING SAMPLE {i+1}/{num_samples_to_test} <<<")
        
        # Pick a random sample
        random_idx = random.randint(0, len(ds) - 1)
        sample = ds[random_idx]

        source_text = sample[config['text_column']]
        human_summary = sample[config['summary_column']]

        # Generate Summary
        print("Generating summary...")
        
        # Handle T5 Prefix
        input_text = source_text
        if config['model_type'] == 't5':
            input_text = "summarize: " + source_text

        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=config['max_input_length'], 
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=150, 
                min_length=30, 
                num_beams=4, 
                length_penalty=2.0, 
                early_stopping=True
            )
        
        model_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display Results
        print("\n" + "="*50)
        print(f"DOCUMENT (Snippet):\n{source_text[:400]}...")
        print("="*50)
        print(f"HUMAN SUMMARY:\n{human_summary}")
        print("-" * 50)
        print(f"MODEL SUMMARY:\n{model_summary}")
        print("="*50)

        # Run Verification
        print("\n🔍 Running Hallucination Analysis...")
        results = verifier.verify(source_text, model_summary)

        for status, claim in results:
            icon = "🔴" if status == "HALLUCINATION" else "🟢"
            print(f"{icon} [{status}]: {claim}")

if __name__ == "__main__":
    main()
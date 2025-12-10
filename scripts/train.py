import sys
import os

# --- PATH SETUP ---
# Add the project root to python path so we can import 'src'
# This ensures the script works whether you run it from root or scripts/ folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from src.utils import load_config, init_wandb
from src.data_loader import load_and_process_data
from src.model_factory import get_model_and_tokenizer

def main():
    # 1. Load Configuration
    # We explicitly look for config.yaml in the project root
    config_path = os.path.join(project_root, "config.yaml")
    config = load_config(config_path)

    # 2. Initialize Logging (W&B)
    # This reads the .env file internally via utils.py
    init_wandb(config)

    # 3. Setup Device (GPU check)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 4. Load Model & Tokenizer (Factory Pattern)
    # This automatically handles the T5 vs BART logic based on config['model_type']
    model, tokenizer = get_model_and_tokenizer(config)
    model.to(device)

    # 5. Load & Preprocess Data
    # This handles the dataset loading (CNN vs XSum) and prefixing
    train_dataset, eval_dataset = load_and_process_data(config, tokenizer)

    # 6. Define Training Arguments
    # We map values directly from the yaml config
    output_dir_abs = os.path.join(project_root, config['output_dir'])
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir_abs,

        # --- Batch Size & Speed ---
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['grad_accum_steps'],
        
        # --- Optimization ---
        learning_rate=float(config['learning_rate']),
        num_train_epochs=config['epochs'],
        weight_decay=0.01,
        
        # --- Evaluation & Saving Strategy ---
        eval_strategy="epoch",     # Evaluate at end of every epoch
        save_strategy="epoch",     # Save checkpoint at end of every epoch
        load_best_model_at_end=True, # Always load the best checkpoint when finished
        metric_for_best_model="eval_loss",
        save_total_limit=2,        # Save space: only keep last 2 checkpoints
        
        # --- Logging (W&B) ---
        logging_steps=100,
        report_to="wandb",         # <--- Critical for your dashboard
        run_name=config['run_name'],

        # --- Hardware Optimization ---
        fp16=torch.cuda.is_available(),  # Use Mixed Precision if GPU is available
        
        # --- Seq2Seq Specifics ---
        predict_with_generate=True,      # Essential for calculating ROUGE/Gen metrics
        generation_max_length=config['max_target_length'],
        remove_unused_columns=False      # Often needed for PEFT/LoRA models
    )

    # 7. Data Collator
    # Handles dynamic padding (faster than padding to max_length manually)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        pad_to_multiple_of=8
    )

    # 8. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9. Start Training
    print(f"\nStarting training for experiment: {config['run_name']}...")
    trainer.train()

    # 10. Save Final Adapter
    print(f"Saving final adapter to {output_dir_abs}...")
    trainer.save_model(output_dir_abs)
    tokenizer.save_pretrained(output_dir_abs)
    
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
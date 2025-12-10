import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

def get_model_and_tokenizer(config):
    """
    Loads the Base Model and Tokenizer, then wraps it with a PEFT/LoRA adapter.
    Handles the specific target modules for T5 vs BART.
    """
    print(f"Loading model: {config['model_name']} ({config['model_type']})")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # 2. Load Base Model
    # We load it in fp16 if CUDA is available to save memory
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch_dtype
    )
    
    # Enable gradient checkpointing to save memory during training
    # (calculates gradients on the fly instead of storing them)
    model.gradient_checkpointing_enable()
    
    # 3. Configure LoRA (PEFT)
    # ⭐️ SWITCH LOGIC: Different models use different names for Attention modules
    if config['model_type'] == 't5':
        # T5 uses 'q' and 'v' in its attention blocks
        target_modules = ["q", "v"]
    elif config['model_type'] == 'bart':
        # BART uses 'q_proj' and 'v_proj'
        target_modules = ["q_proj", "v_proj"]
    else:
        # Fallback default (common for many transformers)
        target_modules = ["q_proj", "v_proj"]
        print(f"⚠️ Warning: Unknown model type '{config['model_type']}'. Defaulting targets to {target_modules}")

    print(f"Applying LoRA to modules: {target_modules}")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=target_modules
    )
    
    # 4. Wrap the model
    model = get_peft_model(model, peft_config)
    
    print("\nModel Architecture (Trainable Parameters):")
    model.print_trainable_parameters()
    
    return model, tokenizer
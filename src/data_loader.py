import sys
import os
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_process_data(config, tokenizer):
    """
    Loads the dataset specified in config and tokenizes it.
    Handles the specific column names for CNN/DailyMail vs XSum.
    """
    print(f"Loading {config['dataset_name']} dataset...")
    
    # 1. Load the raw dataset
    if config['dataset_name'] == 'cnn_dailymail':
        dataset = load_dataset(config['dataset_name'], config['dataset_config'])
    else:
        dataset = load_dataset(config['dataset_name'])

    # 2. Define the Preprocessing Logic
    def preprocess_function(examples):
        # dynamic column names from config (e.g., 'article' vs 'document')
        inputs = examples[config['text_column']]
        targets = examples[config['summary_column']]
        
        # ⭐️ SWITCH LOGIC: T5 needs prefix, BART does not
        if config['model_type'] == 't5':
            inputs = ["summarize: " + doc for doc in inputs]
        else:
            inputs = [doc for doc in inputs]
            
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs, 
            max_length=config['max_input_length'], 
            truncation=True
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=config['max_target_length'], 
                truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 3. Select Subsets
    if config['train_samples'] > 0:
        train_ds = dataset["train"].shuffle(seed=42).select(range(config['train_samples']))
    else:
        train_ds = dataset["train"]
    
    if "validation" in dataset:
        val_split_name = "validation"
    elif "valid" in dataset:
        val_split_name = "valid"
    else:
        val_split_name = "test"
        
    if config['eval_samples'] > 0:
        eval_ds = dataset[val_split_name].shuffle(seed=42).select(range(config['eval_samples']))
    else:
        eval_ds = dataset[val_split_name]

    # 4. Tokenize the datasets
    columns_to_remove = dataset["train"].column_names

    print(f"Tokenizing training data ({len(train_ds)} samples)...")
    tokenized_train = train_ds.map(
        preprocess_function, 
        batched=True,
        remove_columns=columns_to_remove  
    )
    
    print(f"Tokenizing validation data ({len(eval_ds)} samples)...")
    tokenized_eval = eval_ds.map(
        preprocess_function, 
        batched=True,
        remove_columns=columns_to_remove  
    )
    
    return tokenized_train, tokenized_eval
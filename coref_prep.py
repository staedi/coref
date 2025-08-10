#!/usr/bin/env python3
"""
Fine-tuning setup for coreference resolution using MLX-LM
"""

import os
import json
import yaml
import pandas as pd
import argparse
from typing import List, Dict

# First, install required dependencies
def install_dependencies():
    """Install required packages"""
    import subprocess
    import sys
    
    packages = [
        "mlx-lm",
        "pandas",
        "datasets",
        "transformers",
        "torch",
        "pyyaml"
    ]
    
    # for package in packages:
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    subprocess.check_call(["uv", "add", " ".join(packages)])


class CoreferenceDataProcessor:
    """Process coreference resolution data for MLX-LM fine-tuning"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
        
    def create_instruction_format(self, original_text: str, coref_text: str) -> Dict[str, str]:
        """Convert coreference data to instruction format"""
        instruction = (
            "Resolve all coreferences in the following text by replacing pronouns and "
            "descriptive references with their original entities. Maintain the same "
            "meaning and structure while making all references explicit."
        )
        
        return {
            "instruction": instruction,
            "input": original_text,
            "output": coref_text
        }
    
    def process_dataset(self) -> List[Dict[str, str]]:
        """Process the entire dataset into instruction format"""
        processed_data = []
        
        for _, row in self.df.iterrows():
            if pd.notna(row['original']) and pd.notna(row['coref']):
                formatted_item = self.create_instruction_format(
                    row['original'], 
                    row['coref']
                )
                processed_data.append(formatted_item)
        
        return processed_data
    
    def create_chat_format(self, original_text: str, coref_text: str) -> Dict[str, List[Dict[str, str]]]:
        """Create chat format for conversational fine-tuning"""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at coreference resolution. Your task is to resolve all pronouns and descriptive references in text by replacing them with their specific entities."
                },
                {
                    "role": "user", 
                    "content": f"Please resolve all coreferences in this text: {original_text}"
                },
                {
                    "role": "assistant",
                    "content": coref_text
                }
            ]
        }
    
    def save_for_mlx_training(self, output_dir: str = "data", format_type: str = "instruction"):
        """Save processed data in MLX-LM compatible format"""
        os.makedirs(output_dir, exist_ok=True)
        
        if format_type == "instruction":
            processed_data = self.process_dataset()
        elif format_type == "chat":
            processed_data = []
            for _, row in self.df.iterrows():
                if pd.notna(row['original']) and pd.notna(row['coref']):
                    chat_item = self.create_chat_format(row['original'], row['coref'])
                    processed_data.append(chat_item)
        
        # Split data into train/validation (80/20 split)
        split_idx = int(len(processed_data) * 0.8)
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]
        
        # Save as JSONL files
        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "valid.jsonl")
        
        with open(train_path, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        
        with open(val_path, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved {len(train_data)} training samples to {train_path}")
        print(f"Saved {len(val_data)} validation samples to {val_path}")
        
        return train_path, val_path


def create_training_config(model_name:str="meta-llama/llama-3.2-3B-Instruct", iters:int=500, learning_rate:float=1e-5):
    """Create training configuration for MLX-LM"""
    config = {
        "model": model_name,  # The path to the local model directory or Hugging Face repo
        "train": True,  # Whether or not to train (boolean)
        "fine_tune_type": "lora",  # The fine-tuning method: "lora", "dora", or "full"
        "data": "data",  # Directory with {train, valid, test}.jsonl files
        "seed": 0,   # The PRNG seed
        "num_layers": 16,  # Number of layers to fine-tune
        "batch_size": 4, # Minibatch size
        "iters": iters,  # Iterations to train for
        "val_batches": 25,   # Number of validation batches, -1 uses the entire validation set
        "learning_rate": learning_rate, # Adam learning rate
        # "wand": "wandb-project" Whether to report the logs to WandB
        "steps_per_report": 10, # Number of training steps between loss reporting
        "steps_per_eval": 100,  # Number of training steps between validations
        # "resume_adapter_file": None,  # Load path to resume training with the given adapter weights
        "adapter_path": "adapters", # Save/load path for the trained adapter weights
        "save_every": 100,  # Save the model every N iterations
        "test": False,  # Evaluate on the test set after training
        "test_batches": 100,    # Number of test set batches, -1 uses the entire test set
        "max_seq_length": 2048, # Maximum sequence length
        "grad_checkpoint": True,    # Use gradient checkpointing to reduce memory use
        "lora_parameters": {"keys": ["self_attn.q_proj", "self_attn.v_proj"], "rank": 8, "scale": 20.0, "dropout": 0.05}  # LoRA parameters can only be specified in a config file
    }
    
    with open('training_configs.yml', 'w') as f:
        yaml.dump(config, f, indent=2)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune for coreference resolution")
    parser.add_argument("--csv_path", default="data/coreference_dataset.csv", 
                       help="Path to the coreference CSV dataset")
    parser.add_argument("--format", choices=["instruction", "chat"], default="chat",
                       help="Data format for training")
    parser.add_argument("--model_name", default="meta-llama/llama-3.2-3B-Instruct",
                       help="LLM model to use")
    parser.add_argument("--iters", default=500,
                       help="Training iterations")
    parser.add_argument("--learning_rate", default=1e-5,
                       help="Training learning rate")
    parser.add_argument("--install_deps", action="store_true",
                       help="Install required dependencies")
    
    args = parser.parse_args()
    
    if args.install_deps:
        print("Installing dependencies...")
        install_dependencies()
    
    # Process the dataset
    print("Processing coreference dataset...")
    processor = CoreferenceDataProcessor(args.csv_path)
    train_path, val_path = processor.save_for_mlx_training(format_type=args.format)
    
    # Create training configuration
    print("Creating training configuration...")
    config = create_training_config(model_name=args.model_name, iters=args.iters, learning_rate=args.learning_rate)

    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print(f"✓ Dataset processed: {train_path}, {val_path}")
    print(f"✓ Configuration saved: training_config.yml")
    print(f"✓ Format: {args.format}")
    
    print("\nNext steps:")
    print("1. Run the training script:")
    print("   [uv run] python coref_training.py")
    print("\n2. Or use MLX-LM directly:")
    print("   mlx_lm.lora --config training_configs.yml")
    print("\n3. Test the fine-tuned model:")
    print("   [uv run] python inference_coreference.py")

if __name__ == "__main__":
    main()
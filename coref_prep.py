#!/usr/bin/env python3
"""
Fine-tuning setup for coreference resolution using MLX-LM with Gemma3
"""

import os
import json
import pandas as pd
from pathlib import Path
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
        "torch"
    ]
    
    # for package in packages:
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    subprocess.check_call(["uv", "add", " ".join(packages)])

class CoreferenceDataProcessor:
    """Process coreference resolution data for MLX-LM fine-tuning"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
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

# def create_training_config(data_dir: str = 'data', adapter_path: str = 'adapters'):
#     """Create training configuration for MLX-LM"""
#     config = {
#         # "model": "google/gemma-2-2b-it",  # You can change this to other Gemma3 variants
#         "model": "mlx-community/gemma-3-4b-it-4bit",  # You can change this to other Gemma3 variants
#         "data": data_dir,
#         "batch_size": 4,
#         "num_layers": 16,
#         "fine_tune_type": "lora",
#         # "lora_rank": 8,
#         # "lora_alpha": 16,
#         # "lora_dropout": 0.05,
#         "learning_rate": 1e-5,
#         "max_seq_length": 2048,
#         "iters": 500,
#         "val_batches": 25,
#         "steps_per_report": 10,
#         "steps_per_eval": 100,
#         "adapter_path": adapter_path,
#         "save_every": 100,
#         "train": True,
#         "test": False,
#         "test_batches": 100,
#         # "max_tokens_per_batch": 4096,
#         "grad_checkpoint": True
#     }
    
#     with open("training_config.json", "w") as f:
#         json.dump(config, f, indent=2)
    
#     return config

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma3 for coreference resolution")
    parser.add_argument("--csv_path", default="data/coreference_dataset.csv", 
                       help="Path to the coreference CSV dataset")
    parser.add_argument("--format", choices=["instruction", "chat"], default="chat",
                       help="Data format for training")
    parser.add_argument("--model", default="mlx-community/gemma-3n-E4B-it-4bit",
                       help="Gemma3 model variant to use")
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
    
    # # Create training configuration
    # print("Creating training configuration...")
    # config = create_training_config()
    # config["model"] = args.model
    
    # with open("training_config.json", "w") as f:
    #     json.dump(config, f, indent=2)
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print(f"✓ Dataset processed: {train_path}, {val_path}")
    print(f"✓ Configuration saved: training_config.json")
    print(f"✓ Model: {args.model}")
    print(f"✓ Format: {args.format}")
    
    print("\nNext steps:")
    print("1. Run the training script:")
    print("   python coref_training.py")
    print("\n2. Or use MLX-LM directly:")
    print("   mlx_lm.lora --config training_config.yml")
    print("\n3. Test the fine-tuned model:")
    print("   python coref_inf.py")

if __name__ == "__main__":
    main()
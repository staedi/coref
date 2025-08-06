#!/usr/bin/env python3
"""
Training script for coreference resolution using MLX-LM
"""

# import json
import os
import subprocess
import argparse
# from pathlib import Path
# import mlx.core as mx
from mlx_lm import load, generate
# from mlx_lm.utils import load_config

# def run_mlx_training(config_path: str = "training_config.json"):
#     """Run MLX-LM LoRA training"""
    
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Configuration file {config_path} not found")
    
#     with open(config_path, 'r') as f:
#         config = json.load(f)
    
#     # Construct MLX-LM command
#     cmd = [
#         "mlx_lm.lora",
#         "--model", config["model"],
#         "--data", config["data"],
#         "--train",
#         "--batch-size", str(config["batch_size"]),
#         "--num-layers", str(config["num_layers"]),
#         # "--lora-rank", str(config["lora_rank"]), 
#         # "--lora-alpha", str(config["lora_alpha"]),
#         # "--lora-dropout", str(config["lora_dropout"]),
#         "--learning-rate", str(config["learning_rate"]),
#         "--max-seq-length", str(config["max_seq_length"]),
#         "--iters", str(config["iters"]),
#         "--val-batches", str(config["val_batches"]),
#         "--steps-per-report", str(config["steps_per_report"]),
#         "--steps-per-eval", str(config["steps_per_eval"]),
#         "--adapter-path", config["adapter_path"],
#         "--save-every", str(config["save_every"]),
#         # "--max-tokens-per-batch", str(config["max_tokens_per_batch"])
#     ]
    
#     if config.get("grad_checkpoint", False):
#         cmd.append("--grad-checkpoint")
    
#     print("Starting MLX-LM training with command:")
#     print(" ".join(cmd))
#     print("\n" + "="*50)
    
#     # Run training
#     try:
#         result = subprocess.run(cmd, check=True, capture_output=False)
#         print("Training completed successfully!")
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"Training failed with error: {e}")
#         return False

def run_mlx_training(config_path: str = "training_config.yml"):
    """Run MLX-LM LoRA training"""
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    
    # with open(config_path, 'r') as f:
    #     config = json.load(f)
    
    with open(config_path, 'r') as f:
        print(f.read())

    # Construct MLX-LM command
    cmd = [
        "mlx_lm.lora",
        "--config",
        config_path,
        # "--model", config["model"],
        # "--data", config["data"],
        # "--train",
        # "--batch-size", str(config["batch_size"]),
        # "--num-layers", str(config["num_layers"]),
        # # "--lora-rank", str(config["lora_rank"]), 
        # # "--lora-alpha", str(config["lora_alpha"]),
        # # "--lora-dropout", str(config["lora_dropout"]),
        # "--learning-rate", str(config["learning_rate"]),
        # "--max-seq-length", str(config["max_seq_length"]),
        # "--iters", str(config["iters"]),
        # "--val-batches", str(config["val_batches"]),
        # "--steps-per-report", str(config["steps_per_report"]),
        # "--steps-per-eval", str(config["steps_per_eval"]),
        # "--adapter-path", config["adapter_path"],
        # "--save-every", str(config["save_every"]),
        # # "--max-tokens-per-batch", str(config["max_tokens_per_batch"])
    ]
    
    # if config.get("grad_checkpoint", False):
    #     cmd.append("--grad-checkpoint")
    
    print("Starting MLX-LM training with command:")
    print(" ".join(cmd))
    print("\n" + "="*50)
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False

def test_trained_model(adapter_path: str = "adapters", model_name: str = "mlx-community/gemma-3-1b-it-4bit"):
    """Test the trained model with sample coreference resolution"""
    
    print(f"Loading model {model_name} with adapter {adapter_path}...")
    
    try:
        # Load the model with the trained adapter
        model, tokenizer = load(model_name, adapter_path=adapter_path)
        
        # Test samples
        test_samples = [
            "Microsoft announced its quarterly earnings yesterday. The tech giant reported strong growth in cloud services.",
            "Tesla unveiled its new electric vehicle model. The automotive company highlighted the car's innovative features.",
            "OpenAI launched its latest language model. The AI company demonstrated impressive capabilities."
        ]
        
        print("\n" + "="*50)
        print("TESTING TRAINED MODEL")
        print("="*50)
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\nTest {i}:")
            print(f"Input: {sample}")
            
            # Create prompt
            prompt = f"Resolve all coreferences in the following text by replacing pronouns and descriptive references with their original entities: {sample}"
            
            # Generate response
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt,
                max_tokens=200,
                temp=0.1
            )
            
            print(f"Output: {response}")
            print("-" * 50)
    
    except Exception as e:
        print(f"Error testing model: {e}")
        print("Make sure the adapter has been trained and saved correctly.")

def create_inference_script():
    """Create a standalone inference script for the trained model"""
    
    inference_code = '''#!/usr/bin/env python3
"""
Inference script for coreference resolution using fine-tuned Gemma3
"""

import argparse
from mlx_lm import load, generate

def resolve_coreferences(text: str, adapter_path: str = "adapters", 
                        model_name: str = "mlx-community/gemma-3-1b-4t-4bit"):
    """Resolve coreferences in the given text"""
    
    # Load model with adapter
    model, tokenizer = load(model_name, adapter_path=adapter_path)
    
    # Create prompt
    prompt = (
        "Resolve all coreferences in the following text by replacing pronouns and "
        "descriptive references with their original entities. Maintain the same "
        "meaning and structure while making all references explicit: " + text
    )
    
    # Generate response
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        #max_tokens=1000,
        #temp=0.1
    )
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Resolve coreferences using fine-tuned Gemma3")
    parser.add_argument("--text", required=True, help="Text to resolve coreferences in")
    parser.add_argument("--adapter", default="adapters", help="Path to trained adapter")
    parser.add_argument("--model", default="mlx-community/gemma-3-4b-it-4bit", help="Base model name")
    
    args = parser.parse_args()
    
    print("Resolving coreferences...")
    print(f"Input: {args.text}")
    
    try:
        result = resolve_coreferences(args.text, args.adapter, args.model)
        print(f"Output: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("inference_coreference.py", "w") as f:
        f.write(inference_code)
    
    print("Created inference script: inference_coreference.py")

def main():
    parser = argparse.ArgumentParser(description="Train coreference resolution model")
    parser.add_argument("--config", default="training_config.yml", 
                       help="Training configuration file")
    parser.add_argument("--test", action="store_true",
                       help="Test the model after training")
    parser.add_argument("--adapter_path", default="adapters",
                       help="Path where adapters are saved")
    parser.add_argument("--model", default="mlx-community/gemma-3-4b-it-4bit",
                       help="Base model name")
    
    args = parser.parse_args()
    
    # Run training
    print("Starting coreference resolution training...")
    success = run_mlx_training(args.config)
    
    if success:
        print("\n✓ Training completed successfully!")
        
        # Create inference script
        create_inference_script()
        
        if args.test:
            print("\nTesting trained model...")
            test_trained_model(args.adapter_path, args.model)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE!")
        print("="*50)
        print(f"✓ Adapter saved to: {args.adapter_path}")
        print("✓ Inference script created: inference_coreference.py")
        print("\nUsage examples:")
        print(f'python inference_coreference.py --text "Apple announced its earnings. The company performed well."')
        print(f"python -m mlx_lm.generate --model {args.model} --adapter-path {args.adapter_path}")
        
    else:
        print("\n✗ Training failed. Check the error messages above.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Training script for coreference resolution using MLX-LM
"""

import yaml
import os
import subprocess
import argparse
from pathlib import Path
from mlx_lm import load, generate

def run_mlx_training(config_path: str = "training_configs.yml"):
    """Run MLX-LM LoRA training"""
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(config)

    configs = {'model': config['model'], 'adapter_path': config['adapter_path'], 'test': config['test']}

    # Construct MLX-LM command
    cmd = [
        "mlx_lm.lora",
        "--config",
        config_path,
    ]

    print("Starting MLX-LM training with command:")
    print(" ".join(cmd))
    print("\n" + "="*50)

    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("Training completed successfully!")
        success = True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        success = False

    return configs, success


def test_trained_model(model_name:str="meta-llama/llama-3.2-3B-Instruct", adapter_path:str="adapters"):
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
                # max_tokens=1000,
                # temp=0.1
                # verbose=True
            )
            
            print(f"Output: {response}")
            print("-" * 50)
    
    except Exception as e:
        print(f"Error testing model: {e}")
        print("Make sure the adapter has been trained and saved correctly.")


def create_inference_script(model_name:str="meta-llama/llama-3.2-3B-Instruct", adapter_path:str="adapters"):
    """Create a standalone inference script for the trained model"""
    
    inference_code = f'''#!/usr/bin/env python3
"""
Inference script for coreference resolution using fine-tuned model
"""

import argparse
from mlx_lm import load, generate

def resolve_coreferences(text: str, adapter_path: str = {adapter_path}, 
                        model_name: str = {model_name}):
    """Resolve coreferences in the given text"""
    
    # Load model with adapter
    model, tokenizer = load(model_name, adapter_path=adapter_path)
    
    # Create prompt
    prompt = (
        "Resolve all coreferences in the following text by replacing pronouns and "
        "descriptive references with their original entities. Maintain the same "
        "meaning and structure while making all references explicit: \n " + text
        # "Please resolve all coreferences in this text: \n " + text
    )

    if tokenizer.chat_template is not None:
        # messages = [{"role": "system", "content": "You are an expert at coreference resolution. Your task is to resolve all pronouns and descriptive references in text by replacing them with their specific entities."},
        #             {"role": "user", "content": prompt}]
        messages = [{'role':'user','content':prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    # Generate response
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        # max_tokens=1000,
        # temp=0.1,
        # verbose=True
    )
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Resolve coreferences using fine-tuned model")
    parser.add_argument("--text", required=True, help="Text to resolve coreferences in")
    parser.add_argument("--adapter", default={adapter_path}, help="Path to trained adapter")
    parser.add_argument("--model_name", default={model_name}, help="Base model name")
    
    args = parser.parse_args()
    
    print("Resolving coreferences...")
    '''

    inference_code += '''
    print(f"Input: {args.text}")
    
    try:
        result = resolve_coreferences(args.text, args.adapter, args.model_name)
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
    parser.add_argument("--config", default="training_configs.yml", 
                       help="Training configuration file")
    
    args = parser.parse_args()

    prompt = (
        "Resolve all coreferences in the following text by replacing pronouns and "
        "descriptive references with their original entities. Maintain the same "
        "meaning and structure while making all references explicit: \n "
    )

    # Run training
    print("Starting coreference resolution training...")
    configs, success = run_mlx_training(args.config)
    
    if success:
        print("\n✓ Training completed successfully!")

        # Create inference script
        create_inference_script(model_name=configs['model'], adapter_path=configs['adapter_path'])
        
        if configs['test']:
            print("\nTesting trained model...")
            test_trained_model(model_name=configs['model'], adapter_path=configs['adapter_path'])

        print("\n" + "="*50)
        print("TRAINING COMPLETE!")
        print("="*50)
        print(f"✓ Adapter saved to: {configs['adapter_path']}")
        print("\nUsage examples:")
        print(f'[uv run] python inference_coreference.py --text "Apple announced its earnings. The company performed well."')
        print(f'[uv run] python -m mlx_lm.generate --model {configs["model"]} --adapter-path {configs["adapter_path"]} --prompt "{prompt} Apple announced its earnings. The company performed well."')
        
    else:
        print("\n✗ Training failed. Check the error messages above.")

if __name__ == "__main__":
    main()
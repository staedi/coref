#!/usr/bin/env python3
"""
Inference script for coreference resolution using fine-tuned Gemma3
"""

import argparse
from mlx_lm import load, generate

def resolve_coreferences(text: str, adapter_path: str = "adapters", 
                        model_name: str = "mlx-community/gemma-3-4b-it-4bit"):
    """Resolve coreferences in the given text"""
    
    # Load model with adapter
    model, tokenizer = load(model_name, adapter_path=adapter_path)
    
    # Create prompt
    prompt = (
        "Resolve coreferences in the following text by replacing pronouns and "
        "descriptive references with their original entities. Maintain the same "
        "meaning and structure while making all references explicitly: \n " + text
        # "Please resolve all coreferences in this text: \n " + text
    )
    
    if tokenizer.chat_template is not None:
        # messages = [{"role": "system", "content": "You are an expert at coreference resolution. Your task is to resolve all pronouns and descriptive references in text by replacing them with their specific entities."},
        #             {"role": "user", "content": prompt}]
        messages = [{'role':'user','content':prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    # print(messages)

    # Generate response
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        # max_tokens=1000,
        #temp=0.1,
        verbose=True
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

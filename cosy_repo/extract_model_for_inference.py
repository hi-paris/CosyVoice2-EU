#!/usr/bin/env python3
"""
Script to extract model weights from training checkpoints for inference.
Note: this step is not needed in final inference runs when using the final weights, e.g. when using the HF repo.

This script handles both LLM and HiFiGAN models:
- For LLM models: removes epoch/step metadata 
- For HiFiGAN models: extracts only generator weights and removes the 'generator.' prefix

Usage:
    python extract_model_for_inference.py --model llm --input llm-377319.pt --output llm-377319-clean.pt
    python extract_model_for_inference.py --model hifigan --input hifigan_averaged.pt --output hift-latest.pt
"""

import argparse
import torch
import os
import sys


def extract_llm_weights(input_path, output_path):
    """Extract LLM weights by removing training metadata."""
    print(f"Loading LLM checkpoint from: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu')
    
    # Count original keys
    original_keys = len(ckpt)
    
    # Remove unwanted training metadata keys
    metadata_keys = ['epoch', 'step']
    for key in metadata_keys:
        ckpt.pop(key, None)
    
    print(f"Removed {original_keys - len(ckpt)} metadata keys")
    print(f"Saving clean LLM weights to: {output_path}")
    torch.save(ckpt, output_path)
    print(f"Successfully saved {len(ckpt)} weight keys.")


def extract_hifigan_weights(input_path, output_path):
    """Extract HiFiGAN generator weights by removing discriminator and 'generator.' prefix."""
    print(f"Loading HiFiGAN checkpoint from: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu')
    
    # Count original keys
    original_keys = len(ckpt)
    
    # Filter to only generator keys and remove the 'generator.' prefix
    generator_weights = {}
    generator_count = 0
    discriminator_count = 0
    other_count = 0
    
    for key, value in ckpt.items():
        if key.startswith('generator.'):
            # Remove the 'generator.' prefix for inference compatibility
            new_key = key.replace('generator.', '')
            generator_weights[new_key] = value
            generator_count += 1
        elif key.startswith('discriminator.'):
            discriminator_count += 1
            # Skip discriminator weights - not needed for inference
        else:
            # Handle any other keys (epoch, step, etc.)
            if key not in ['epoch', 'step']:
                generator_weights[key] = value
            other_count += 1
    
    print(f"Original checkpoint structure:")
    print(f"  - Generator weights: {generator_count}")
    print(f"  - Discriminator weights: {discriminator_count}")
    print(f"  - Other keys: {other_count}")
    print(f"  - Total keys: {original_keys}")
    
    if generator_count == 0:
        print("ERROR: No generator weights found in checkpoint!")
        print("Available keys (first 20):")
        for i, key in enumerate(sorted(ckpt.keys())):
            if i < 20:
                print(f"  {key}")
            else:
                print(f"  ... and {len(ckpt) - 20} more")
                break
        return False
    
    print(f"Saving HiFT-compatible weights to: {output_path}")
    torch.save(generator_weights, output_path)
    print(f"Successfully saved {len(generator_weights)} HiFT weight keys.")
    
    # Verify key structure
    print(f"\nExtracted weight key examples (first 10):")
    for i, key in enumerate(sorted(generator_weights.keys())):
        if i < 10:
            print(f"  {key}")
        else:
            print(f"  ... and {len(generator_weights) - 10} more")
            break
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract model weights for inference')
    parser.add_argument('--model', required=True, choices=['llm', 'hifigan'], 
                       help='Type of model to extract weights from')
    parser.add_argument('--input', required=True, 
                       help='Input checkpoint path')
    parser.add_argument('--output', required=True,
                       help='Output path for cleaned weights')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite output file if it exists')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file does not exist: {args.input}")
        sys.exit(1)
    
    # Check output file doesn't exist (unless force)
    if os.path.exists(args.output) and not args.force:
        print(f"ERROR: Output file already exists: {args.output}")
        print("Use --force to overwrite")
        sys.exit(1)
    
    # Extract weights based on model type
    try:
        if args.model == 'llm':
            extract_llm_weights(args.input, args.output)
        elif args.model == 'hifigan':
            success = extract_hifigan_weights(args.input, args.output)
            if not success:
                sys.exit(1)
        
        print(f"\n✅ Successfully extracted {args.model} weights!")
        print(f"You can now use {args.output} for inference.")
        
    except Exception as e:
        print(f"ERROR: Failed to extract weights: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

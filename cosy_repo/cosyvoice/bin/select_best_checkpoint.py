#!/usr/bin/env python3
"""
Smart checkpoint selector for CosyVoice2 LoRA training.
Prioritizes new CosyVoice2-compatible checkpoints, falls back to validation loss ranking.
"""

import argparse
import os
import sys
import yaml
import shutil
from pathlib import Path


def find_best_cosyvoice2_checkpoint(src_path):
    """Find the best CosyVoice2-compatible checkpoint by scanning validation yamls."""
    src_path = Path(src_path)
    best_checkpoint = None
    best_loss = float('inf')
    best_epoch = None
    
    print(f"Scanning for CosyVoice2 checkpoints in: {src_path}")
    
    # Scan for validation YAML files
    for yaml_file in src_path.glob("*.yaml"):
        if yaml_file.name.startswith(('train', 'init')):
            continue
            
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            loss = data['loss_dict']['loss']
            epoch = data['epoch']
            
            # Look for corresponding CosyVoice2 checkpoint
            cosyvoice2_path = src_path / f"epoch_merged/epoch_{epoch}_merged/cosyvoice2.pt"
            if cosyvoice2_path.exists() and loss < best_loss:
                best_checkpoint = cosyvoice2_path
                best_loss = loss
                best_epoch = epoch
                print(f"Found CosyVoice2 checkpoint: epoch {epoch}, loss {loss:.6f}")
                
        except (KeyError, TypeError, yaml.YAMLError):
            continue
    
    return best_checkpoint, best_epoch, best_loss


def main():
    parser = argparse.ArgumentParser(description='Select best checkpoint for CosyVoice2')
    parser.add_argument('--src_path', required=True, help='Source training directory')
    parser.add_argument('--dst_model', required=True, help='Destination checkpoint path')
    args = parser.parse_args()
    
    # Try to find CosyVoice2-compatible checkpoint
    best_checkpoint, best_epoch, best_loss = find_best_cosyvoice2_checkpoint(args.src_path)
    
    if best_checkpoint:
        print(f"Selected CosyVoice2 checkpoint from epoch {best_epoch} (loss: {best_loss:.6f})")
        shutil.copy2(best_checkpoint, args.dst_model)
        print(f"Copied to: {args.dst_model}")
        return 0
    else:
        print("No CosyVoice2-compatible checkpoints found")
        return 1


if __name__ == '__main__':
    sys.exit(main())

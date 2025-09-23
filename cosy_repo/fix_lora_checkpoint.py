#!/usr/bin/env python3
"""
Script to convert LoRA merged checkpoint to CosyVoice2-compatible format.

This script takes a merged LoRA checkpoint (HuggingFace format) and combines it
with the CosyVoice2-specific layers to create a proper inference checkpoint.
"""

import torch
import os
import sys

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Install with: pip install safetensors")

def convert_lora_to_cosyvoice2(merged_model_path, original_lora_checkpoint, output_path):
    """
    Convert a merged LoRA model to CosyVoice2-compatible format.
    
    Args:
        merged_model_path: Path to the merged HuggingFace model (model.safetensors or pytorch_model.bin)
        original_lora_checkpoint: Path to original LoRA checkpoint (epoch_X_whole.pt) 
        output_path: Path for the output CosyVoice2 checkpoint
    """
    
    print(f"Loading merged backbone from: {merged_model_path}")
    
    # Load the merged backbone weights
    if merged_model_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            print("ERROR: safetensors library required to load .safetensors files")
            print("Install with: pip install safetensors")
            sys.exit(1)
        merged_weights = load_file(merged_model_path)
    else:
        merged_weights = torch.load(merged_model_path, map_location='cpu')
    
    print(f"Loaded {len(merged_weights)} keys from merged model")
    
    print(f"Loading original LoRA checkpoint from: {original_lora_checkpoint}")
    
    # Load the original LoRA checkpoint to get CosyVoice2-specific layers
    original_checkpoint = torch.load(original_lora_checkpoint, map_location='cpu')
    print(f"Loaded {len(original_checkpoint)} keys from original checkpoint")
    
    # Create the new checkpoint structure
    new_checkpoint = {}
    
    # Copy epoch/step info if available
    if 'epoch' in original_checkpoint:
        new_checkpoint['epoch'] = original_checkpoint['epoch']
        print(f"Keeping epoch: {original_checkpoint['epoch']}")
    if 'step' in original_checkpoint:
        new_checkpoint['step'] = original_checkpoint['step']
        print(f"Keeping step: {original_checkpoint['step']}")
    
    print("\nProcessing weights...")
    
    # First, add all merged backbone weights with correct prefixes
    backbone_count = 0
    for merged_key, merged_value in merged_weights.items():
        # Convert HuggingFace key to CosyVoice2 key
        cosyvoice_key = f"llm.model.{merged_key}"
        new_checkpoint[cosyvoice_key] = merged_value
        backbone_count += 1
        if backbone_count <= 5:  # Show first few for verification
            print(f"  Adding backbone: {merged_key} -> {cosyvoice_key}")
    
    print(f"Added {backbone_count} backbone weights")
    
    # Then, add CosyVoice2-specific layers from original checkpoint
    cosyvoice_count = 0
    skipped_count = 0
    
    for key, value in original_checkpoint.items():
        if key in ['epoch', 'step']:
            continue
            
        # Skip backbone keys (we already have them from merged model)
        if key.startswith('llm.model.model.'):
            continue
            
        # Skip LoRA-specific keys
        if any(lora_pattern in key for lora_pattern in ['base_model', 'lora_A', 'lora_B', 'modules_to_save']):
            skipped_count += 1
            if skipped_count <= 5:  # Show first few skipped
                print(f"  Skipping LoRA key: {key}")
            continue
            
        # Keep CosyVoice2-specific layers
        new_checkpoint[key] = value
        cosyvoice_count += 1
        if cosyvoice_count <= 10:  # Show first few
            print(f"  Keeping CosyVoice2 layer: {key}")
    
    print(f"Added {cosyvoice_count} CosyVoice2-specific weights")
    print(f"Skipped {skipped_count} LoRA-specific keys")
    
    # Verify we have the essential CosyVoice2 components
    essential_keys = [
        'llm_embedding.weight',
        'llm_decoder.weight', 
        'llm_decoder.bias',
        'speech_embedding.weight'
    ]
    
    missing_keys = [key for key in essential_keys if key not in new_checkpoint]
    if missing_keys:
        print(f"\n⚠️  WARNING: Missing essential CosyVoice2 keys: {missing_keys}")
        print("This may cause inference issues!")
    else:
        print(f"\n✅ All essential CosyVoice2 components found")
    
    print(f"\nSaving converted checkpoint to: {output_path}")
    torch.save(new_checkpoint, output_path)
    
    print("\n🎉 Conversion completed!")
    print(f"Original checkpoint keys: {len(original_checkpoint)}")
    print(f"Merged model keys: {len(merged_weights)}")  
    print(f"New checkpoint keys: {len(new_checkpoint)}")
    
    # Print key statistics
    backbone_keys = [k for k in new_checkpoint.keys() if k.startswith('llm.model.model.')]
    cosyvoice_keys = [k for k in new_checkpoint.keys() if not k.startswith('llm.model.model.') and k not in ['epoch', 'step']]
    
    print(f"\nFinal checkpoint structure:")
    print(f"  - Backbone transformer keys: {len(backbone_keys)}")
    print(f"  - CosyVoice2-specific keys: {len(cosyvoice_keys)}")
    print(f"  - Total keys: {len(new_checkpoint)}")
    
    return True


def print_usage():
    """Print usage instructions"""
    print("Usage: python fix_lora_checkpoint.py <merged_model_path> <original_checkpoint> <output_path>")
    print()
    print("Arguments:")
    print("  merged_model_path     : Path to the merged model (HuggingFace dir or .safetensors/.bin file)")
    print("  original_checkpoint   : Path to the original LoRA checkpoint (.pt file)")  
    print("  output_path          : Path for the output CosyVoice2-compatible checkpoint")
    print()
    print("Examples:")
    print("  # Using HuggingFace directory from epoch_merged")
    print("  python fix_lora_checkpoint.py epoch_merged/epoch_3_merged/ epoch_3_whole.pt llm_fixed.pt")
    print()
    print("  # Using safetensors from epoch_merged")
    print("  python fix_lora_checkpoint.py epoch_merged/epoch_3_merged/model.safetensors epoch_3_whole.pt llm_fixed.pt")
    print()
    print("  # Using pytorch_model.bin from epoch_merged")  
    print("  python fix_lora_checkpoint.py epoch_merged/epoch_3_merged/pytorch_model.bin epoch_3_whole.pt llm_fixed.pt")


def convert_lora_checkpoint(merged_model_path, original_checkpoint_path, output_path):
    """Convert single file merged checkpoint to CosyVoice2 format."""
    try:
        # Load the merged model
        print(f"Loading merged model from: {merged_model_path}")
        if merged_model_path.endswith('.safetensors'):
            from safetensors import safe_open
            merged_state_dict = {}
            with safe_open(merged_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged_state_dict[key] = f.get_tensor(key)
        else:
            merged_state_dict = torch.load(merged_model_path, map_location='cpu')
        
        print(f"✅ Loaded merged model with {len(merged_state_dict)} parameters")
        
        # Load original checkpoint for CosyVoice2-specific components
        print(f"Loading original checkpoint: {original_checkpoint_path}")
        original_checkpoint = torch.load(original_checkpoint_path, map_location='cpu')
        print(f"✅ Loaded original checkpoint with {len(original_checkpoint)} keys")
        
        # Create the final checkpoint
        final_checkpoint = {}
        
        # Add merged backbone with correct prefix
        print("Remapping backbone weights to llm.model.model.* format...")
        backbone_count = 0
        for k, v in merged_state_dict.items():
            if k.startswith('model.'):
                # HuggingFace format has model.* keys, CosyVoice2 expects llm.model.model.*
                new_key = f"llm.{k}"  # model.* becomes llm.model.*
                final_checkpoint[new_key] = v
                backbone_count += 1
        print(f"✅ Added {backbone_count} backbone parameters")
        
        # Add CosyVoice2-specific components from original
        cosyvoice_components = [
            'speech_embedding',
            'llm_decoder', 
            'llm_embedding',  # This was missing!
            'flow',
            'hift'
        ]
        
        cosyvoice_count = 0
        for component in cosyvoice_components:
            component_keys = [k for k in original_checkpoint.keys() if k.startswith(component)]
            for key in component_keys:
                final_checkpoint[key] = original_checkpoint[key]
                cosyvoice_count += 1
        print(f"✅ Added {cosyvoice_count} CosyVoice2-specific parameters")
        
        # Add metadata
        final_checkpoint['epoch'] = original_checkpoint.get('epoch', 0)
        final_checkpoint['step'] = original_checkpoint.get('step', 0)
        final_checkpoint['lr'] = original_checkpoint.get('lr', 0.0)
        
        # Save the final checkpoint
        print(f"Saving final checkpoint to: {output_path}")
        torch.save(final_checkpoint, output_path)
        
        print(f"✅ Successfully created checkpoint with {len(final_checkpoint)} total keys")
        print(f"   - Backbone (LLM) parameters: {backbone_count}")
        print(f"   - CosyVoice2-specific parameters: {cosyvoice_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_hf_dir_to_cosyvoice2(hf_dir, original_checkpoint_path, output_path):
    """Convert HuggingFace directory (like epoch_merged/epoch_X_merged/) to CosyVoice2 format.
    
    This is similar to select_best_checkpoint.py but handles the full CosyVoice2 requirements.
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("❌ Error: transformers not available. Please install: pip install transformers")
        return False
    
    try:
        print(f"Loading HuggingFace model from: {hf_dir}")
        model = AutoModelForCausalLM.from_pretrained(hf_dir, trust_remote_code=True)
        merged_state_dict = model.state_dict()
        print(f"✅ Loaded HF model with {len(merged_state_dict)} parameters")
        
        # Load original checkpoint for CosyVoice2-specific components
        print(f"Loading original checkpoint: {original_checkpoint_path}")
        original_checkpoint = torch.load(original_checkpoint_path, map_location='cpu')
        print(f"✅ Loaded original checkpoint with {len(original_checkpoint)} keys")
        
        # Create the final checkpoint
        final_checkpoint = {}
        
        # Add merged backbone with correct prefix
        print("Remapping backbone weights to llm.model.model.* format...")
        backbone_count = 0
        for k, v in merged_state_dict.items():
            new_key = f"llm.model.{k}"  # Add the llm.model. prefix
            final_checkpoint[new_key] = v
            backbone_count += 1
        print(f"✅ Added {backbone_count} backbone parameters")
        
        # Add CosyVoice2-specific components from original
        cosyvoice_components = [
            'speech_embedding',
            'llm_decoder', 
            'llm_embedding',  # This was missing!
            'flow',
            'hift'
        ]
        
        cosyvoice_count = 0
        for component in cosyvoice_components:
            component_keys = [k for k in original_checkpoint.keys() if k.startswith(component)]
            for key in component_keys:
                final_checkpoint[key] = original_checkpoint[key]
                cosyvoice_count += 1
        print(f"✅ Added {cosyvoice_count} CosyVoice2-specific parameters")
        
        # Add metadata
        # final_checkpoint['epoch'] = original_checkpoint.get('epoch', 0)
        # final_checkpoint['step'] = original_checkpoint.get('step', 0)
        # final_checkpoint['lr'] = original_checkpoint.get('lr', 0.0)
        
        # Save the final checkpoint
        print(f"Saving final checkpoint to: {output_path}")
        torch.save(final_checkpoint, output_path)
        
        print(f"✅ Successfully created checkpoint with {len(final_checkpoint)} total keys")
        print(f"   - Backbone (LLM) parameters: {backbone_count}")
        print(f"   - CosyVoice2-specific parameters: {cosyvoice_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting HF directory: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) != 4:
        print_usage()
        sys.exit(1)
    
    merged_model_path = sys.argv[1]
    original_checkpoint_path = sys.argv[2] 
    output_path = sys.argv[3]
    
    print(f"Converting LoRA merged checkpoint to CosyVoice2 format...")
    print(f"Source merged model: {merged_model_path}")
    print(f"Source original checkpoint: {original_checkpoint_path}")
    print(f"Output checkpoint: {output_path}")
    
    # Check if merged_model_path is a HuggingFace directory or a file
    if os.path.isdir(merged_model_path):
        print("Detected HuggingFace model directory, using optimized conversion...")
        success = convert_hf_dir_to_cosyvoice2(merged_model_path, original_checkpoint_path, output_path)
    else:
        print("Detected single file, using file-based conversion...")
        success = convert_lora_checkpoint(merged_model_path, original_checkpoint_path, output_path)
    
    if success:
        print(f"\\n✅ Successfully created CosyVoice2-compatible checkpoint: {output_path}")
        print("You can now use this checkpoint for inference!")
    else:
        print("❌ Conversion failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

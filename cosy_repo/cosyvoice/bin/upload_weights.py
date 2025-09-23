#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
from huggingface_hub import HfApi, login


def get_args():
    parser = argparse.ArgumentParser(description='Upload trained model weights to HuggingFace Hub')
    parser.add_argument('--exp_dir',
                        type=str,
                        required=True,
                        help='Path to experiment directory containing trained models (e.g., exp/cosyvoice2)')
    parser.add_argument('--hf_repo_id',
                        type=str,
                        required=True,
                        help='Hugging Face repository ID (e.g., username/model-name)')
    parser.add_argument('--hf_token',
                        type=str,
                        default=None,
                        help='Hugging Face token for authentication (or set HF_TOKEN env var)')
    parser.add_argument('--train_engine',
                        type=str,
                        default='torch_ddp',
                        help='Training engine used (default: torch_ddp)')
    parser.add_argument('--models',
                        nargs='+',
                        default=['llm', 'flow', 'hifigan'],
                        help='Models to upload (default: llm flow hifigan)')
    args = parser.parse_args()
    return args


def find_best_checkpoint(model_dir):
    """Find the best checkpoint in the model directory."""
    # <export_folder>/cosyvoice2/llm/torch_ddp
    # we want --> llm from the path

    # Look for averaged model first (check both naming conventions)
    model_name = os.path.basename(os.path.dirname(model_dir))  # e.g., llm, flow, hifigan

    averaged_path1 = os.path.join(model_dir, f'{model_name}_averaged.pt')
    averaged_path2 = os.path.join(model_dir, 'averaged_model.pt')
    
    if os.path.exists(averaged_path1):
        return averaged_path1
    else:
        print(f"WARNING: Averaged model not found at {averaged_path1}, checking {averaged_path2}")
    if os.path.exists(averaged_path2):
        return averaged_path2
    else:
        print(f"WARNING: Averaged model not found at {averaged_path2}, looking for final checkpoint")
    
    # Look for final checkpoint
    final_path = os.path.join(model_dir, 'final_model.pt')
    if os.path.exists(final_path):
        return final_path
    else:
        print(f"WARNING: Final model not found at {final_path}, looking for any checkpoint")

    # Look for any .pt file with highest epoch number
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'epoch' in f]
    if pt_files:
        # Sort by epoch number (handle both 'epoch_10.pt' and 'epoch_10_whole.pt' formats)
        def extract_epoch_num(filename):
            try:
                # Handle both epoch_N.pt and epoch_N_whole.pt patterns
                if '_whole.pt' in filename:
                    return int(filename.split('_')[1])
                elif 'epoch_' in filename:
                    return int(filename.split('_')[1].split('.')[0])
                else:
                    return 0
            except (ValueError, IndexError):
                return 0
        
        pt_files.sort(key=extract_epoch_num, reverse=True)
        return os.path.join(model_dir, pt_files[0])
    
    # Fallback to any .pt file
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if pt_files:
        return os.path.join(model_dir, pt_files[0])
    
    return None


def main():
    args = get_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    
    # Authenticate with HF Hub
    try:
        if args.hf_token:
            login(token=args.hf_token)
        elif os.getenv('HF_TOKEN'):
            login(token=os.getenv('HF_TOKEN'))
        else:
            logging.warning("No HF token provided. Make sure you're logged in with 'huggingface-cli login'")
        
        api = HfApi()
        
        # Upload model weights
        uploaded_files = []
        
        for model in args.models:
            model_dir = os.path.join(args.exp_dir, model, args.train_engine)
            
            if not os.path.exists(model_dir):
                logging.warning(f"Model directory not found: {model_dir}")
                continue
            
            # Find best checkpoint
            checkpoint_path = find_best_checkpoint(model_dir)
            if not checkpoint_path:
                logging.warning(f"No checkpoint found in {model_dir}")
                continue
            
            # Map hifigan to hift for consistency with CosyVoice2
            upload_name = 'hift.pt' if model == 'hifigan' else f'{model}.pt'
            
            logging.info(f"Uploading {model} checkpoint: {checkpoint_path} -> {upload_name}")
            
            try:
                api.upload_file(
                    path_or_fileobj=checkpoint_path,
                    path_in_repo=upload_name,
                    repo_id=args.hf_repo_id,
                    repo_type="model"
                )
                uploaded_files.append(upload_name)
                logging.info(f"Successfully uploaded {upload_name}")
                
            except Exception as e:
                logging.error(f"Failed to upload {upload_name}: {e}")
        
        if uploaded_files:
            logging.info(f"Successfully uploaded {len(uploaded_files)} model files to https://huggingface.co/{args.hf_repo_id}")
            logging.info(f"Uploaded files: {', '.join(uploaded_files)}")
        else:
            logging.warning("No files were uploaded")
            
    except Exception as e:
        logging.error(f"Failed to upload to Hugging Face Hub: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

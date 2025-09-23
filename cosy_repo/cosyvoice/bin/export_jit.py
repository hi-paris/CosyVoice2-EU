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

from __future__ import print_function

import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import sys
import torch
from huggingface_hub import HfApi, login
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../..'.format(ROOT_DIR))
sys.path.append('{}/../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import logging


def get_args():
    parser = argparse.ArgumentParser(description='export your model for deployment')
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='local path')
    parser.add_argument('--hf_repo_id',
                        type=str,
                        default=None,
                        help='Hugging Face repository ID (e.g., username/model-name) to push exported models')
    parser.add_argument('--hf_token',
                        type=str,
                        default=None,
                        help='Hugging Face token for authentication (or set HF_TOKEN env var)')
    parser.add_argument('--final',
                        action='store_true',
                        help='Use final version of the model (llm.pt and flow.pt instead of llm-original.pt and flow-original.pt)')
    args = parser.parse_args()
    print(args)
    return args


def get_optimized_script(model, preserved_attrs=[]):
    script = torch.jit.script(model)
    if preserved_attrs != []:
        script = torch.jit.freeze(script, preserved_attrs=preserved_attrs)
    else:
        script = torch.jit.freeze(script)
    script = torch.jit.optimize_for_inference(script)
    return script


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    torch._C._jit_set_fusion_strategy([('STATIC', 1)])
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    try:
        model = CosyVoice(args.model_dir)
        print('successfully load cosyvoice model')
    except Exception:
        try:
            model = CosyVoice2(args.model_dir, final=args.final)
            print('successfully load cosyvoice 2 model')
        except Exception:
            raise TypeError('no valid model_type!')

    if not isinstance(model, CosyVoice2):
        # 1. export llm text_encoder
        llm_text_encoder = model.model.llm.text_encoder
        script = get_optimized_script(llm_text_encoder)
        script.save('{}/llm.text_encoder.fp32.zip'.format(args.model_dir))
        script = get_optimized_script(llm_text_encoder.half())
        script.save('{}/llm.text_encoder.fp16.zip'.format(args.model_dir))
        logging.info('successfully export llm_text_encoder')

        # 2. export llm llm
        llm_llm = model.model.llm.llm
        script = get_optimized_script(llm_llm, ['forward_chunk'])
        script.save('{}/llm.llm.fp32.zip'.format(args.model_dir))
        script = get_optimized_script(llm_llm.half(), ['forward_chunk'])
        script.save('{}/llm.llm.fp16.zip'.format(args.model_dir))
        logging.info('successfully export llm_llm')

        # 3. export flow encoder
        flow_encoder = model.model.flow.encoder
        script = get_optimized_script(flow_encoder)
        script.save('{}/flow.encoder.fp32.zip'.format(args.model_dir))
        script = get_optimized_script(flow_encoder.half())
        script.save('{}/flow.encoder.fp16.zip'.format(args.model_dir))
        logging.info('successfully export flow_encoder')
    else:
        # 3. export flow encoder
        flow_encoder = model.model.flow.encoder
        script = get_optimized_script(flow_encoder)
        script.save('{}/flow.encoder.fp32.zip'.format(args.model_dir))
        script = get_optimized_script(flow_encoder.half())
        script.save('{}/flow.encoder.fp16.zip'.format(args.model_dir))
        logging.info('successfully export flow_encoder')
    
    # Push to Hugging Face Hub if requested
    if args.hf_repo_id:
        try:
            # Authenticate with HF Hub
            if args.hf_token:
                login(token=args.hf_token)
            elif os.getenv('HF_TOKEN'):
                login(token=os.getenv('HF_TOKEN'))
            else:
                logging.warning("No HF token provided. Make sure you're logged in with 'huggingface-cli login'")
            
            api = HfApi()
            
            # Upload all exported files
            exported_files = []
            for file in os.listdir(args.model_dir):
                if file.endswith('.zip'):
                    exported_files.append(os.path.join(args.model_dir, file))
            
            if exported_files:
                logging.info(f"Uploading {len(exported_files)} JIT model files to {args.hf_repo_id}")
                for file_path in exported_files:
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=os.path.basename(file_path),
                        repo_id=args.hf_repo_id,
                        repo_type="model"
                    )
                logging.info(f"Successfully uploaded JIT models to https://huggingface.co/{args.hf_repo_id}")
            else:
                logging.warning("No JIT model files found to upload")
                
        except Exception as e:
            logging.error(f"Failed to upload to Hugging Face Hub: {e}")


if __name__ == '__main__':
    main()

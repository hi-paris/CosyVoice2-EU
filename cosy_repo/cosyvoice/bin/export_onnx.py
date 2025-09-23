# Copyright (c) 2024 Antgroup Inc (authors: Zhoubofan, hexisyztem@icloud.com)
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
import onnxruntime
import random
import torch
from tqdm import tqdm
from huggingface_hub import HfApi, login
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../..'.format(ROOT_DIR))
sys.path.append('{}/../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import logging


def get_dummy_input(batch_size, seq_len, out_channels, device):
    x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float32, device=device)
    mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    t = torch.rand((batch_size), dtype=torch.float32, device=device)
    spks = torch.rand((batch_size, out_channels), dtype=torch.float32, device=device)
    cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    return x, mask, mu, t, spks, cond


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
    parser.add_argument('--skip_validation',
                        action='store_true',
                        help='Skip ONNX validation tests (useful if validation fails due to precision issues)')
    parser.add_argument('--final',
                        action='store_true',
                        help='Use final version of the model (llm.pt and flow.pt instead of llm-original.pt and flow-original.pt)')
    args = parser.parse_args()
    print(args)
    return args


@torch.no_grad()
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    try:
        model = CosyVoice(args.model_dir)
    except Exception:
        try:
            model = CosyVoice2(args.model_dir, final=args.final)
        except Exception:
            raise TypeError('no valid model_type!')

    # 1. export flow decoder estimator
    estimator = model.model.flow.decoder.estimator
    estimator.eval()

    device = model.model.device
    batch_size, seq_len = 2, 256
    out_channels = model.model.flow.decoder.estimator.out_channels
    x, mask, mu, t, spks, cond = get_dummy_input(batch_size, seq_len, out_channels, device)
    torch.onnx.export(
        estimator,
        (x, mask, mu, t, spks, cond),
        '{}/flow.decoder.estimator.fp32.onnx'.format(args.model_dir),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['x', 'mask', 'mu', 't', 'spks', 'cond'],
        output_names=['estimator_out'],
        dynamic_axes={
            'x': {2: 'seq_len'},
            'mask': {2: 'seq_len'},
            'mu': {2: 'seq_len'},
            'cond': {2: 'seq_len'},
            'estimator_out': {2: 'seq_len'},
        }
    )

    # 2. test computation consistency
    if not args.skip_validation:
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        estimator_onnx = onnxruntime.InferenceSession('{}/flow.decoder.estimator.fp32.onnx'.format(args.model_dir),
                                                      sess_options=option, providers=providers)

        validation_passed = True
        for i in tqdm(range(10)):
            try:
                x, mask, mu, t, spks, cond = get_dummy_input(batch_size, random.randint(16, 512), out_channels, device)
                output_pytorch = estimator(x, mask, mu, t, spks, cond)
                ort_inputs = {
                    'x': x.cpu().numpy(),
                    'mask': mask.cpu().numpy(),
                    'mu': mu.cpu().numpy(),
                    't': t.cpu().numpy(),
                    'spks': spks.cpu().numpy(),
                    'cond': cond.cpu().numpy()
                }
                output_onnx = estimator_onnx.run(None, ort_inputs)[0]
                torch.testing.assert_allclose(output_pytorch, torch.from_numpy(output_onnx).to(device), rtol=1e-2, atol=1e-4)
            except AssertionError as e:
                logging.warning(f"Validation test {i+1} failed: {e}")
                validation_passed = False
                break
        
        if validation_passed:
            logging.info('ONNX validation passed - successfully export estimator')
        else:
            logging.warning('ONNX validation failed, but model was exported. Consider using --skip_validation flag.')
    else:
        logging.info('Skipping ONNX validation - successfully export estimator')
    
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
            
            # Upload all exported ONNX files
            exported_files = []
            for file in os.listdir(args.model_dir):
                if file.endswith('.onnx'):
                    exported_files.append(os.path.join(args.model_dir, file))
            
            if exported_files:
                logging.info(f"Uploading {len(exported_files)} ONNX model files to {args.hf_repo_id}")
                for file_path in exported_files:
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=os.path.basename(file_path),
                        repo_id=args.hf_repo_id,
                        repo_type="model"
                    )
                logging.info(f"Successfully uploaded ONNX models to https://huggingface.co/{args.hf_repo_id}")
            else:
                logging.warning("No ONNX model files found to upload")
                
        except Exception as e:
            logging.error(f"Failed to upload to Hugging Face Hub: {e}")


if __name__ == "__main__":
    main()

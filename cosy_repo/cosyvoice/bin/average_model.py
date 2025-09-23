# Copyright (c) 2020 Mobvoi Inc (Di Wu)
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

import os
import argparse
import glob
import re

import yaml
import torch


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--val_best',
                        action="store_true",
                        help='averaged model')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    val_scores = []
    if args.val_best:
        yamls = glob.glob('{}/*.yaml'.format(args.src_path))
        yamls = [
            f for f in yamls
            if not (os.path.basename(f).startswith('train')
                    or os.path.basename(f).startswith('init'))
        ]
        for y in yamls:
            with open(y, 'r') as f:
                dic_yaml = yaml.load(f, Loader=yaml.BaseLoader)
                loss = float(dic_yaml['loss_dict']['loss'])
                epoch = int(dic_yaml['epoch'])
                step = int(dic_yaml['step'])
                tag = dic_yaml['tag']
                val_scores += [[epoch, step, loss, tag]]
        sorted_val_scores = sorted(val_scores,
                                   key=lambda x: x[2],
                                   reverse=False)
        print("best val (epoch, step, loss, tag) = " +
              str(sorted_val_scores[:args.num]))
        
        # Find checkpoint files for the actual best validation scores
        path_list = []
        for score in sorted_val_scores[:args.num]:
            epoch = score[0]
            step = score[1]
            
            # Try to find the exact step checkpoint first
            step_pt = os.path.join(args.src_path, f'epoch_{epoch}_step_{step}.pt')
            if os.path.exists(step_pt):
                path_list.append(step_pt)
            else:
                # Fallback to whole.pt if the exact step doesn't exist
                whole_pt = os.path.join(args.src_path, f'epoch_{epoch}_whole.pt')
                if os.path.exists(whole_pt):
                    path_list.append(whole_pt)
                else:
                    # Last resort: find latest step checkpoint for this epoch
                    step_pts = glob.glob(os.path.join(args.src_path, f'epoch_{epoch}_step_*.pt'))
                    if step_pts:
                        def get_step(fname):
                            match = re.search(r'step_(\d+)\.pt', fname)
                            return int(match.group(1)) if match else 0
                        latest_pt = max(step_pts, key=get_step)
                        path_list.append(latest_pt)
                    else:
                        print(f"Warning: No checkpoint found for epoch {epoch}, step {step}, skipping.")
        
        if not path_list:
            raise RuntimeError("No valid checkpoints found for averaging!")
        
        print(f"Selected {len(path_list)} best checkpoints for averaging:")
    for p in path_list:
        print(f"  {p}")
    
    avg = {}
    num = len(path_list)  # Use actual number of found checkpoints
    for path in path_list:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        for k in states.keys():
            if k not in ['step', 'epoch']:
                if k not in avg.keys():
                    avg[k] = states[k].clone()
                else:
                    avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(args.dst_model))
    torch.save(avg, args.dst_model)


if __name__ == '__main__':
    main()

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
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist
import deepspeed
import traceback
from copy import deepcopy
from peft import PeftModel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: W&B not available. Install with: pip install wandb")

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.losses import DPOLoss
from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)


def _sanitize_for_wandb(x):
    """Recursively convert non-serializable objects into JSON-safe representations.
    W&B expects config values to be basic types; this prevents objects like
    functools.partial from being passed directly and causing init errors.
    """
    # primitives
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    # dict -> sanitize keys and values
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            # convert key to str to be safe
            try:
                out[str(k)] = _sanitize_for_wandb(v)
            except Exception:
                out[str(k)] = str(v)
        return out
    # list/tuple -> sanitize elements
    if isinstance(x, (list, tuple)):
        return [_sanitize_for_wandb(v) for v in x]
    # other types -> fallback to string representation
    try:
        return str(x)
    except Exception:
        return repr(x)

def _str2bool(x): return str(x).lower() in ("1","true","t","yes","y")


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--ref_model', required=False, help='ref model used in dpo')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser.add_argument('--use_wandb',
                        action='store_true',
                        default=False,
                        help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--wandb_project',
                        default='cosyvoice',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name',
                        default=None,
                        help='W&B run name (auto-generated if not provided)')
    parser.add_argument('--slurm_job_id',
                        default=None,
                        help='SLURM job ID for tracking')
    parser.add_argument('--tokenizer_add_specials',
                    type=str, choices=['true','false'], default=None,
                    help='Whether to add extra CosyVoice special tokens')
    parser.add_argument('--max_frames_in_batch',
                        type=int,
                        default=None,
                        help='Override max_frames_in_batch from config')
    parser.add_argument('--grad_checkpoint', action='store_true', help='Enable gradient checkpointing for HF backbone')
    parser.add_argument('--lora_enable', action='store_true', help='Enable LoRA fine-tuning for HF backbone instead of full SFT')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj',
                        help='Comma separated list of module name substrings to apply LoRA to')
    # Resume controls (torch_ddp only for now)
    parser.add_argument('--resume', action='store_true', help='Resume training (HF backbones only)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Explicit checkpoint path to resume from')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='Required when --resume is set to keep the same W&B run')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # gan train has some special initialization logic
    gan = True if args.model == 'hifigan' else False

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')

    # Adding additional overrides here, e.g. handling of tokens for custom backbones
    extra_overrides = {}
    if args.tokenizer_add_specials is not None:
        extra_overrides['get_tokenizer'] = {'add_additional_specials': _str2bool(args.tokenizer_add_specials)}
    # allow runtime override for batch.max_frames_in_batch
    if getattr(args, 'max_frames_in_batch', None) is not None:
        extra_overrides['batch'] = {'max_frames_in_batch': args.max_frames_in_batch}

    try:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={**override_dict, 'qwen_pretrain_path': args.qwen_pretrain_path, **extra_overrides})
    except Exception:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={**override_dict, **extra_overrides})
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))

    # Init env for ddp
    init_distributed(args)
    rank = int(os.environ.get('RANK', 0))

    # Initialize wandb only on rank 0
    if args.use_wandb and rank == 0 and WANDB_AVAILABLE:
        # Enforce wandb_run_id when resuming (user requested behavior)
        if args.resume and not args.wandb_run_id:
            raise ValueError("--resume was set but --wandb_run_id is missing. Provide the existing W&B run id to continue.")
        # Auto-generate run name if not provided
        if args.wandb_run_name is None:
            run_name = f"{args.model}"
            if args.slurm_job_id:
                run_name += f"-job{args.slurm_job_id}"
            args.wandb_run_name = run_name
        
        try:
            # Preflight diagnostics to help debug mysterious functools.partial error
            logging.info(f"[W&B Preflight] wandb version: {getattr(wandb, '__version__', 'unknown')}")
            logging.info(f"[W&B Preflight] wandb.init callable: {wandb.init}")
            logging.info(f"[W&B Preflight] SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')} PWD={os.getcwd()}")
            # Explicitly clear any accidental env-provided JSON config that could hold partials
            for env_key in ['WANDB_CONFIG', 'WANDB_CONFIG_JSON']:
                if env_key in os.environ:
                    logging.info(f"[W&B Preflight] Unsetting {env_key} to avoid side-effects")
                    os.environ.pop(env_key, None)
            # Initialize wandb with a STRICTLY PRIMITIVE config to avoid objects like functools.partial
            def _primitive_filter(d):
                out = {}
                if not isinstance(d, dict):
                    return out
                for k, v in d.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        out[str(k)] = v
                return out

            base_conf = _primitive_filter(configs.get('train_conf', {}))
            # Add selected primitive metrics from nested known sections if present
            for section in ['optim_conf', 'sched_conf', 'batch']:
                sub = configs.get(section, {}) if isinstance(configs.get(section, {}), dict) else {}
                for k, v in sub.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        base_conf[f"{section}.{k}"] = v

            # Safely get max_frames_in_batch: configs['batch'] might be a functools.partial (from hyperpyyaml factory)
            batch_conf = configs.get('batch', None)
            if isinstance(batch_conf, dict):
                cfg_max_frames = batch_conf.get('max_frames_in_batch')
            else:
                if batch_conf is not None and not isinstance(batch_conf, dict):
                    logging.debug(f"[W&B Config] 'batch' entry is type {type(batch_conf)}; skipping field extraction.")
                cfg_max_frames = None
            effective_max_frames = args.max_frames_in_batch if args.max_frames_in_batch is not None else cfg_max_frames

            runtime_conf = {
                'model_type': args.model,
                'train_engine': args.train_engine,
                'slurm_job_id': args.slurm_job_id,
                'checkpoint': args.checkpoint,
                'model_dir': args.model_dir,
                'dpo': args.dpo,
                'use_amp': args.use_amp,
                'train_data': args.train_data,
                'cv_data': args.cv_data,
                'max_frames_in_batch': effective_max_frames,
            }
            safe_config = {**base_conf, **_sanitize_for_wandb(runtime_conf)}

            # Further simplified robust initialization: strip everything to isolate error source
            init_attempts = []
            error_log_path = os.path.join(os.getcwd(), 'wandb_init_error_traceback.log')

            def log_trace(label, exc):
                tb = traceback.format_exc()
                logging.debug(tb)
                try:
                    with open(error_log_path, 'a') as f:
                        f.write(f"===== {label} ({type(exc).__name__}) =====\n")
                        f.write(str(exc) + "\n")
                        f.write(tb + "\n")
                except Exception:
                    pass

            # Attempt 1: resume-aware init if requested
            try:
                if args.resume and args.wandb_run_id:
                    run = wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume='must')
                else:
                    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)
                init_attempts.append('proj_name_success_primary')
            except Exception as e1:
                init_attempts.append(f'proj_name_primary_fail:{type(e1).__name__}')
                logging.warning(f"W&B init (project+name primary) failed: {e1}")
                log_trace('project_name_primary', e1)
                # Attempt 2: retry with project only (even more minimal)
                try:
                    if args.resume and args.wandb_run_id:
                        run = wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume='must')
                    else:
                        run = wandb.init(project=args.wandb_project)
                    init_attempts.append('proj_only_fallback_success')
                except Exception as e2:
                    init_attempts.append(f'proj_only_fallback_fail:{type(e2).__name__}')
                    logging.warning(f"W&B init (project only fallback) failed: {e2}")
                    log_trace('project_only_fallback', e2)
                    # Attempt 3: disable metadata collection via env + settings-less fallback
                    os.environ['WANDB_DISABLE_GIT'] = 'true'
                    os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', 'online')
                    try:
                        if args.resume and args.wandb_run_id:
                            run = wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume='must', mode=os.environ['WANDB_MODE'])
                        else:
                            run = wandb.init(project=args.wandb_project, mode=os.environ['WANDB_MODE'])
                        init_attempts.append('final_basic_success')
                    except Exception as e3:
                        init_attempts.append(f'final_basic_fail:{type(e3).__name__}')
                        logging.warning(f"W&B init (final basic) failed: {e3}")
                        log_trace('final_basic', e3)
                        # Give up: re-raise so outer handler disables wandb
                        raise

            # Push sanitized config only if run active
            if wandb.run is not None:
                try:
                    wandb.config.update(safe_config, allow_val_change=True)
                except Exception as e:
                    logging.warning(f"Failed to update W&B config post-init: {e}")
                    logging.debug(f"Safe config keys: {list(safe_config.keys())}")
            logging.info(f"W&B initialization attempts: {init_attempts}")

            if args.use_wandb and rank == 0 and WANDB_AVAILABLE:
                # Save the raw YAML file as an artifact-like file in the run
                try:
                    wandb.save(args.config)  # uploads the file so you can download/compare later
                except Exception as e:
                    logging.warning(f"Could not attach config file to W&B: {e}")
            
            # Log SLURM output files if available
            if args.slurm_job_id:
                slurm_out_file = f"slurm-{args.slurm_job_id}.out"
                slurm_err_file = f"slurm-{args.slurm_job_id}.err"
                if os.path.exists(slurm_out_file):
                    wandb.save(slurm_out_file)
                if os.path.exists(slurm_err_file):
                    wandb.save(slurm_err_file)
            logging.info("W&B initialization completed successfully")
        except Exception as e:
            tb = traceback.format_exc()
            logging.warning(f"W&B initialization failed: {e}. Continuing without W&B logging.")
            logging.error(tb)
            # dump traceback to file for later inspection
            try:
                with open(os.path.join(os.getcwd(), 'wandb_init_error_traceback.log'), 'a') as f:
                    f.write("===== W&B init failure =====\n")
                    f.write(str(e) + "\n")
                    f.write(tb + "\n")
            except Exception:
                pass
            args.use_wandb = False
    elif args.use_wandb and not WANDB_AVAILABLE:
        logging.warning("W&B requested but not available. Install with: pip install wandb")
        args.use_wandb = False

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan, args.dpo)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)
    # If AMP requested under torch_ddp, reflect in dtype metadata for logging/logic
    if args.use_amp and args.train_engine == 'torch_ddp':
        configs['train_conf']['dtype'] = 'fp16'

    # Tensorboard summary
    writer = init_summarywriter(args)

    # load checkpoint
    if args.dpo is True:
        configs[args.model].forward = configs[args.model].forward_dpo
    model = configs[args.model]
    # Optional: apply LoRA / gradient checkpointing only for HF backbone (llm)
    if args.model == 'llm':
        # Detect HF backbone
        from cosyvoice.llm import llm as llm_mod
        hf_backbone = None
        try:
            if hasattr(model, 'llm') and hasattr(model.llm, 'model'):
                hf_backbone = model.llm.model
        except Exception:
            hf_backbone = None
        
        # LoRA injection (do this first, before gradient checkpointing)
        if args.lora_enable and hf_backbone is not None:
            try:
                from peft import LoraConfig, get_peft_model
                target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()]
                lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules,
                                      lora_dropout=args.lora_dropout, bias='none', task_type='CAUSAL_LM')
                model.llm.model = get_peft_model(hf_backbone, lora_cfg)
                trainable = sum(p.numel() for p in model.llm.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.llm.model.parameters())
                logging.info(f'LoRA enabled. Trainable params: {trainable}/{total} ({100.0*trainable/total:.2f}%)')
                # Update backbone reference after LoRA wrapping
                hf_backbone = model.llm.model
            except ImportError:
                logging.error('peft not installed. Install with pip install peft to use LoRA.')
                hf_backbone = None
            except Exception as e:
                logging.error(f'Failed to apply LoRA: {e}')
                hf_backbone = None
        
        # Gradient checkpointing (after LoRA if enabled)
        if args.grad_checkpoint and hf_backbone is not None:
            try:
                # For PEFT models, access the base model for gradient checkpointing
                model_for_gc = hf_backbone.base_model if hasattr(hf_backbone, 'base_model') else hf_backbone
                if hasattr(model_for_gc, 'gradient_checkpointing_enable'):
                    if hasattr(model_for_gc, 'config') and hasattr(model_for_gc.config, 'use_cache'):
                        model_for_gc.config.use_cache = False
                    model_for_gc.gradient_checkpointing_enable()
                    logging.info('Enabled gradient checkpointing for HF backbone')
            except Exception as e:
                logging.warning(f'Failed to enable gradient checkpointing: {e}')
    def _auto_find_latest_ckpt(dir_path: str):
        try:
            if not os.path.isdir(dir_path):
                return None
            cands = []
            for name in os.listdir(dir_path):
                p = os.path.join(dir_path, name)
                if os.path.isfile(p) and name.endswith('.pt'):
                    cands.append((os.path.getmtime(p), p))
            if not cands:
                return None
            cands.sort()
            return cands[-1][1]
        except Exception:
            return None

    start_step, start_epoch = 0, -1
    resume_path = None
    if args.resume:
        # Prefer explicit path, else try latest from model_dir
        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
            resume_path = args.resume_from_checkpoint
        else:
            resume_path = _auto_find_latest_ckpt(args.model_dir)
        if resume_path is None:
            logging.warning(f"--resume was set but no checkpoint found to resume from in {args.model_dir}.")
        else:
            logging.info(f"Resuming from checkpoint: {resume_path}")
            try:
                state_dict = torch.load(resume_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
                if isinstance(state_dict, dict):
                    start_step = int(state_dict.get('step', start_step))
                    start_epoch = int(state_dict.get('epoch', start_epoch))
            except Exception as e:
                logging.warning(f"Failed to load resume checkpoint {resume_path}: {e}")
    elif args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            if isinstance(state_dict, dict):
                start_step = int(state_dict.get('step', start_step))
                start_epoch = int(state_dict.get('epoch', start_epoch))
        else:
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)
    scheduler.set_step(start_step)
    if scheduler_d is not None:
        scheduler_d.set_step(start_step)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)

    # DPO related
    if args.dpo is True:
        ref_model = deepcopy(configs[args.model])
        state_dict = torch.load(args.ref_model, map_location='cpu')
        ref_model.load_state_dict(state_dict, strict=False)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
        # NOTE maybe it is not needed to wrap ref_model as ddp because its parameter is not updated
        ref_model = wrap_cuda_model(args, ref_model)
    else:
        ref_model, dpo_loss = None, None

    # Get executor
    executor = Executor(gan=gan, ref_model=ref_model, dpo_loss=dpo_loss)
    executor.step = start_step

    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    print('start step {} start epoch {}'.format(start_step, start_epoch))

    # Start training loop
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        if gan is True:
            executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                                        writer, info_dict, scaler, group_join)
        else:
            executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=ref_model)
        dist.destroy_process_group(group_join)

        # After each epoch, if LoRA enabled, only snapshot adapter weights (no merge) for inspection
        if args.model == 'llm' and args.lora_enable and int(os.environ.get('RANK', 0)) == 0:
            try:
                actual_model = model.module if hasattr(model, 'module') else model
                if hasattr(actual_model, 'llm') and hasattr(actual_model.llm, 'model') and isinstance(actual_model.llm.model, PeftModel):
                    base_dir = os.path.join(args.model_dir, 'epoch_lora')
                    os.makedirs(base_dir, exist_ok=True)
                    adapter_dir = os.path.join(base_dir, f'epoch_{epoch}_adapters')
                    if not os.path.exists(adapter_dir):
                        actual_model.llm.model.save_pretrained(adapter_dir)
                        logging.info(f'Saved LoRA adapters snapshot to {adapter_dir}')
            except Exception as e:
                logging.warning(f'LoRA epoch adapter snapshot failed: {e}')

        # Safe per-epoch MERGED export (rank 0) -> on CPU! (to avoid touching the training graph)
        if args.model == 'llm' and args.lora_enable and int(os.environ.get('RANK', 0)) == 0:
            try:
                actual = model.module if hasattr(model, 'module') else model
                peft = getattr(getattr(actual, 'llm', None), 'model', None)
                if isinstance(peft, PeftModel):
                    export_dir = os.path.join(args.model_dir, 'epoch_merged')
                    os.makedirs(export_dir, exist_ok=True)
                    this_out = os.path.join(export_dir, f'epoch_{epoch}_merged')

                    # Work on a CPU copy so we don't touch the training graph
                    with torch.no_grad():
                        peft_cpu = deepcopy(peft).to('cpu')
                        peft_cpu.eval()
                        base = peft_cpu.merge_and_unload()   # returns the base HF model with weights merged
                        
                        # Save HF-style (so you can from_pretrained later)
                        base.save_pretrained(this_out)
                        
                        # Create CosyVoice2-compatible checkpoint for direct inference
                        logging.info(f"Creating CosyVoice2-compatible checkpoint for epoch {epoch}")
                        cosyvoice2_checkpoint = {}
                        
                        # Add merged backbone with correct CosyVoice2 key prefix
                        base_state_dict = base.state_dict()
                        for k, v in base_state_dict.items():
                            cosyvoice2_key = f"llm.model.{k}"  # Add llm.model. prefix
                            cosyvoice2_checkpoint[cosyvoice2_key] = v
                        
                        # Get CosyVoice2-specific components from current training state
                        current_state = {}
                        if hasattr(model, 'module'):
                            current_state = model.module.state_dict()
                        else:
                            current_state = model.state_dict()
                        
                        # Add CosyVoice2-specific components
                        cosyvoice_components = ['speech_embedding', 'llm_decoder', 'llm_embedding', 'flow', 'hift']
                        for component in cosyvoice_components:
                            for key, value in current_state.items():
                                if key.startswith(component):
                                    cosyvoice2_checkpoint[key] = value.cpu()
                        
                        # Add metadata
                        # cosyvoice2_checkpoint['epoch'] = epoch
                        # cosyvoice2_checkpoint['step'] = executor.step
                        # cosyvoice2_checkpoint['lr'] = optimizer.param_groups[0]['lr']
                        
                        # Save CosyVoice2-compatible checkpoint
                        cosyvoice2_path = os.path.join(this_out, 'cosyvoice2.pt')
                        torch.save(cosyvoice2_checkpoint, cosyvoice2_path)
                        logging.info(f"Saved CosyVoice2-compatible checkpoint: {cosyvoice2_path}")
                        
                        # Also save as PyTorch format for HF compatibility  
                        torch.save(base.state_dict(), os.path.join(this_out, 'llm.pt'))
                        
                        # Save tokenizer if you want a self-contained export
                        try:
                            from transformers import AutoTokenizer
                            tok = AutoTokenizer.from_pretrained(args.qwen_pretrain_path or actual.llm.tokenizer.name_or_path)
                            tok.save_pretrained(this_out)
                        except Exception:
                            pass
                        del base, peft_cpu
            except Exception as e:
                logging.warning(f'Per-epoch merged export failed: {e}')


    # Final LoRA merge checkpoint for inference
    if args.model == 'llm' and args.lora_enable:
        info_dict['final_lora_merge'] = True
        save_model(model, 'final_merged', info_dict)

    # Finalize wandb
    if args.use_wandb and rank == 0 and WANDB_AVAILABLE:
        try:
            if wandb.run is not None:
                wandb.finish()
                logging.info("W&B run finished successfully")
        except Exception as e:
            logging.warning(f"W&B finalization error: {e}")


if __name__ == '__main__':
    main()

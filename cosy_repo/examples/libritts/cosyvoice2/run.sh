#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=7
stop_stage=8
models="llm flow"  # default models to train, can be overridden by command line
train_hours=1000  # default training hours, can be overridden by command line
gpu_partition="GPU"  # default GPU partition name, can be overridden by command line
backbone="blanken" # "blanken", "hf:Qwen/Qwen2.5-0.5B", "hf:Qwen/Qwen3-0.6B", "hf:utter-project/EuroLLM-1.7B-Instruct"
grad_checkpoint="${COSY_GRAD_CHECKPOINT:-false}"
lora_enable="${COSY_LORA_ENABLE:-false}"
language="${COSY_LANGUAGE:-"mix"}" # Language code (e.g., "fr", "de", or "mix" for both)

# Resume flags (HF backbones only). Defaults are safe no-op.
resume_continue="false"
wandb_run_id=""
resume_from_checkpoint=""

# Set Hugging Face repository ID here (optional)
# NOTE: if not set, data will not be uploaded to Hugging Face
HF_REPO_ID="Luka512/CosyVoice2-0.5B-EU"


# STAGES OVERVIEW:
# -1: Data Download
#  0: Data Preparation
#  1: Extract Speaker Embedding
#  2: Extract Speech Token
#  3: Prepare Parquet Data
#  5: Train Model
#  6: Average Model
#  7: Export Models
#  8: Upload Final Model to Hugging Face


# backbone to identifier map
# blanken -> bl, hf:Qwen/Qwen2.5-0.5B -> q2, hf:Qwen/Qwen3-0.6B -> q3, hf:utter-project/EuroLLM-1.7B-Instruct -> eu

declare -A backbone_to_id_map
backbone_to_id_map=(
  ["blanken"]="bl"
  ["hf:Qwen/Qwen2.5-0.5B"]="q2"
  ["hf:Qwen/Qwen3-0.6B"]="q3"
  ["hf:utter-project/EuroLLM-1.7B-Instruct"]="eu"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --stage)
      stage="$2"
      shift 2
      ;;
    --stop_stage)
      stop_stage="$2"
      shift 2
      ;;
    --models)
      models="$2"
      shift 2
      ;;
    --num_workers)
      num_workers="$2"
      shift 2
      ;;
    --train_engine)
      train_engine="$2"
      shift 2
      ;;
    --train_hours)
      train_hours="$2"
      shift 2
      ;;
    --gpu_partition)
      gpu_partition="$2"
      shift 2
      ;;
    --backbone)
      backbone="$2"
      shift 2
      ;;
    --max_frames_in_batch)
      max_frames_in_batch="$2"
      shift 2
      ;;
    --grad_checkpoint)
      grad_checkpoint="$2"
      shift 2
      ;;
    --lora_enable)
      lora_enable="$2"
      shift 2
      ;;
    --language)
      language="$2"
      shift 2
      ;;
    --continue)
      resume_continue="$2"
      shift 2
      ;;
    --wandb_run_id)
      wandb_run_id="$2"
      shift 2
      ;;
    --resume_from_checkpoint)
      resume_from_checkpoint="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--stage N] [--stop_stage N] [--models 'model1 model2 ...'] [--num_workers N] [--train_engine engine] [--train_hours N] [--gpu_partition NAME] [--backbone NAME] [--max_frames_in_batch N] [--grad_checkpoint true|false] [--lora_enable true|false] [--language LANG]"
      echo "  --models: Space-separated list of models to train (e.g., 'llm', 'flow', 'hifigan', 'llm flow hifigan')"
      exit 1
      ;;
  esac
done

# Sanitize backbone string for folder naming
if [ -n "$backbone" ]; then
  backbone_sanitized=$(echo "$backbone" | sed 's#[/:]#-#g')
else
  backbone_sanitized="nobackbone"
fi


data_url=www.openslr.org/resources/60
data_dir="/tsi/hi-paris/tts/Luka/data"
folder_name="tts_dataset_combined_${train_hours}h_${language}"

cosy_data_dir="/tsi/hi-paris/tts/Luka/data/cosyvoice2_combined_${train_hours}h_${language}"

pretrained_model_dir=/home/infres/horstmann-24/TTS2/cosy_repo/pretrained_models/CosyVoice2-0.5B

export_dir="/tsi/hi-paris/tts/Luka/exp-${train_hours}h-${backbone_sanitized}-lora_${lora_enable}"

# export WANDB_DIR
export WANDB_DIR="/tsi/hi-paris/tts/Luka/wandb"


parts="train dev test"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Data Download"
  for part in $parts; do
    local/download_and_untar.sh ${data_dir} ${data_url} ${part}
  done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in $parts; do
    mkdir -p $cosy_data_dir/$x
    python local/prepare_data.py --src_dir $data_dir/$folder_name/$x --des_dir $cosy_data_dir/$x
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in $cosy_data_dir/$x dir"
  for x in $parts; do
    tools/extract_embedding.py --dir $cosy_data_dir/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in $cosy_data_dir/$x dir"
  for x in $parts; do
    tools/extract_speech_token.py --dir $cosy_data_dir/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in $parts; do
    mkdir -p $cosy_data_dir/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 8 \
      --src_dir $cosy_data_dir/$x \
      --des_dir $cosy_data_dir/$x/parquet
  done
fi

# train llm
# export CUDA_VISIBLE_DEVICES="0,1" # here we specify the GPUs we want to use
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=${num_workers:-2}  # default to 2 if not specified
prefetch=100
train_engine=${train_engine:-torch_ddp}  # default to torch_ddp if not specified

# W&B settings
use_wandb=true
wandb_project="${WANDB_PROJECT:-CosyVoice2-EU}"  # Use environment variable if set, otherwise default
wandb_run_name="${models}_${train_hours}h-${backbone_sanitized}-lora_${lora_enable}_${language}"  # Default run name based on parameters
echo "W&B run name:    $wandb_run_name"
echo "Gradient checkpoint: $grad_checkpoint"
echo "LoRA enable: $lora_enable"

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat $cosy_data_dir/train/parquet/data.list > $cosy_data_dir/train.data.list
  cat $cosy_data_dir/dev/parquet/data.list > $cosy_data_dir/dev.data.list

  # Decide backbone path + tokenizer extras from --backbone
  QWEN_PRETRAIN_PATH=""
  ADD_SPECIALS="true"   # default

  case "$backbone" in
    blanken)
      # original CosyVoice-BlankEN
      QWEN_PRETRAIN_PATH="$pretrained_model_dir/CosyVoice-BlankEN"
      ADD_SPECIALS="true"
      ;;
    hf:*)
      # arbitrary HF repo
      QWEN_PRETRAIN_PATH="${backbone#hf:}"
      ADD_SPECIALS="false"
      ;;
    local:*)
      # local model dir
      QWEN_PRETRAIN_PATH="${backbone#local:}"
      # keep it conservative for non-BlankEN
      ADD_SPECIALS="false"
      ;;
    *)
      echo "Unknown --backbone value: $backbone"
      echo "Use one of: blanken | hf:<repo> | local:<path>"
      exit 1
      ;;
  esac

  echo "Using backbone: $backbone"
  echo "  qwen_pretrain_path = $QWEN_PRETRAIN_PATH"
  echo "  add_additional_specials = $ADD_SPECIALS"
  if [[ "$resume_continue" == "true" ]]; then
    echo "Resume requested (HF only). wandb_run_id=${wandb_run_id}, resume_from_checkpoint=${resume_from_checkpoint:-auto}"
  fi


  # NOTE will update llm/hift training later
  for model in $models; do # models specified via --models parameter or default to llm # can be 'llm', 'flow', 'hifigan', etc.
    # Prepare wandb args
    wandb_args=""
    if [ "$use_wandb" = true ]; then
      wandb_args="--use_wandb"
      if [ -n "$wandb_project" ]; then
        wandb_args="$wandb_args --wandb_project $wandb_project"
      fi
      if [ -n "$wandb_run_name" ]; then
        wandb_args="$wandb_args --wandb_run_name $wandb_run_name"
      fi
      if [ -n "$SLURM_JOB_ID" ]; then
        wandb_args="$wandb_args --slurm_job_id $SLURM_JOB_ID"
      fi
    fi

    # choose checkpoint per component/backbone
    CKPT_ARG=""
    if [ "$model" = "llm" ]; then
      if [ "$backbone" = "blanken" ]; then
        CKPT_ARG="--checkpoint $pretrained_model_dir/llm.pt"
      else
        CKPT_ARG=""   # start LLM head fresh when using arbitrary HF models
      fi
    elif [ "$model" = "flow" ]; then
      CKPT_ARG="--checkpoint $pretrained_model_dir/flow.pt"
    elif [ "$model" = "hifigan" ]; then
      CKPT_ARG="--checkpoint $pretrained_model_dir/hifigan.pt"
    fi

    
    # build extra runtime args for train.py
    EXTRA_TRAIN_ARGS=""
    if [ -n "${max_frames_in_batch:-}" ]; then
      EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --max_frames_in_batch $max_frames_in_batch"
    fi

    # Add resume args for HF backbone only
    RESUME_ARGS=""
    if [[ "$backbone" == hf:* || "$backbone" == local:* ]]; then
      if [[ "$resume_continue" == "true" ]]; then
        if [[ -z "$wandb_run_id" ]]; then
          echo "Error: --continue true requires --wandb_run_id for HF backbones." >&2
          exit 2
        fi
        RESUME_ARGS="--resume --wandb_run_id $wandb_run_id"
        if [[ -n "$resume_from_checkpoint" ]]; then
          RESUME_ARGS="$RESUME_ARGS --resume_from_checkpoint $resume_from_checkpoint"
        fi
      fi
    fi

    python -m torch.distributed.run --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500}" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2.yaml \
      --train_data $cosy_data_dir/train.data.list \
      --cv_data $cosy_data_dir/dev.data.list \
      --qwen_pretrain_path $QWEN_PRETRAIN_PATH \
      --tokenizer_add_specials $ADD_SPECIALS \
      --model $model \
      $CKPT_ARG \
      --model_dir $export_dir/cosyvoice2/$model/$train_engine \
      --tensorboard_dir /tsi/hi-paris/tts/Luka/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer \
      $( [[ "$grad_checkpoint" == "true" ]] && echo "--grad_checkpoint" ) \
      $( [[ "$lora_enable" == "true" ]] && echo "--lora_enable" ) \
      $wandb_args $EXTRA_TRAIN_ARGS $RESUME_ARGS
  done
fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then

  backbone_id=${backbone_to_id_map[$backbone]}

  for model in $models; do
    model_train_dir=$export_dir/cosyvoice2/$model/$train_engine
    echo "source path: $model_train_dir"
    
    # Check if training checkpoints exist
    if [ ! -d "$model_train_dir" ]; then
      echo "Warning: Model directory $model_train_dir does not exist. Trying language upper."

      exp_dir_with_last_lang_upper="/tsi/hi-paris/tts/Luka/exp-${train_hours}h-${backbone_sanitized}-lora_${lora_enable}_${language^^}"
      model_train_dir="$exp_dir_with_last_lang_upper/cosyvoice2/$model/$train_engine"

      if [ ! -d "$model_train_dir" ]; then
        echo "Warning: Model directory $model_train_dir does not exist. Skipping $model."
        continue
      fi
    fi

    # recompute decode_checkpoint after potential fallback that updates model_train_dir
    decode_checkpoint=$model_train_dir/${model}-${train_hours}-averaged-${backbone_id}-${language^^}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"

    if [ "$lora_enable" == "true" ] && [ "$model" = "llm" ]; then
      echo "LoRA enabled for $model: selecting best checkpoint"
      
      # Try to use the enhanced select_best_checkpoint.py first (looks for CosyVoice2 checkpoints)
      if python cosyvoice/bin/select_best_checkpoint.py --src_path $model_train_dir --dst_model $decode_checkpoint 2>/dev/null; then
        echo "Successfully selected best CosyVoice2-compatible checkpoint"
      else
        echo "No CosyVoice2 checkpoints found, using legacy conversion..."
        # Fall back to our fix_lora_checkpoint.py script for legacy checkpoints
        if [[ -f "/home/infres/horstmann-24/TTS2/cosy_repo/fix_lora_checkpoint.py" ]]; then
          python /home/infres/horstmann-24/TTS2/cosy_repo/fix_lora_checkpoint.py \
            "$model_train_dir/epoch_merged/epoch_2_merged/" \
            "$model_train_dir/epoch_2_whole.pt" \
            "$decode_checkpoint" || {
            echo "Error: Legacy conversion failed for $model"; continue; }
        else
          echo "Error: No conversion method available for $model"; continue
        fi
      fi
    else
      # Legacy averaging path (unchanged) for non-LoRA or non-LLM components
      epoch_count=$(find $model_train_dir -name "epoch_*_whole.pt" | wc -l)
      if [ $epoch_count -eq 0 ]; then
        echo "Warning: No epoch checkpoints found in $model_train_dir. Skipping averaging for $model."
        continue
      fi
      actual_num=$(( epoch_count < average_num ? epoch_count : average_num ))
      echo "Found $epoch_count epoch checkpoints, averaging top $actual_num (val_best)."
      python cosyvoice/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $model_train_dir \
        --num ${actual_num} \
        --val_best || { echo "Averaging failed for $model"; continue; }
    fi

    # copy averaged model to /home/infres/horstmann-24/TTS2/cosy_repo/pretrained_models/CosyVoice2-0.5B folder
    cp $decode_checkpoint /home/infres/horstmann-24/TTS2/cosy_repo/pretrained_models/CosyVoice2-0.5B/
    echo "Copied averaged model to /home/infres/horstmann-24/TTS2/cosy_repo/pretrained_models/CosyVoice2-0.5B/"
  done
fi


final_model_dir=$export_dir/cosyvoice2/final_model
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export models"
  
  # commenting this below out because we can just upload the full final folder to HF and have everything covered, so that we don't have to duplicate logic

  # if [ -n "$HF_REPO_ID" ]; then
  #   echo "Will upload to HuggingFace Hub: $HF_REPO_ID"
    
  #   # Upload model weights for each trained model (upload script will automatically find best checkpoint)
  #   for model in $models; do
  #     echo "Uploading best available checkpoint for $model..."
  #     python cosyvoice/bin/upload_weights.py \
  #       --exp_dir $export_dir/cosyvoice2 \
  #       --hf_repo_id $HF_REPO_ID \
  #       --train_engine $train_engine \
  #       --models $model
  #   done
  # fi
  
  # Create a unified model directory for export (works for both HF upload and local export)
  mkdir -p $final_model_dir
  echo "Creating model directory for export: $final_model_dir"
  
  # Copy trained models (averaged if available, otherwise skip - export scripts will use pretrained)

  # if model is hifigan, we then also have to run extract_model_for_inference.py to:
    # Extract only the generator weights (328 keys) from your complete HiFiGAN checkpoint (472 keys)
    # Remove the generator. prefix to match the expected weight names
    # Discard discriminator weights (144 keys) that are not needed for inference
    # Save the clean weights as hift.pt that CosyVoice2 can load
  for model in $models; do
    model_train_dir=$export_dir/cosyvoice2/$model/$train_engine
    averaged_checkpoint=$model_train_dir/${model}_averaged.pt
    if [ -f "$averaged_checkpoint" ]; then
      echo "Copying trained $model to export directory..."
      cp $averaged_checkpoint $final_model_dir/${model}.pt
    else
      echo "No trained $model found, using pretrained version for export"
    fi

    if [ "$model" == "hifigan" ]; then
      echo "Extracting HiFiGAN weights for inference..."
      python ../../../extract_model_for_inference.py \
        --model hifigan \
        --input $final_model_dir/${model}.pt \
        --output $final_model_dir/hift.pt \
        --force
    fi
  done
  
  # Export optimized models
  

  # Export required files from pretrained folder
  # copy from pretrained model dir: CosyVoice-BlankEN folder, campplus.onnx, speech_tokenizer_v2.onnx, cosyvoice2.yaml
  echo "Copying pretrained files to final export directory..."
  if [[ "$backbone" == "blanken" ]]; then
    cp -r $pretrained_model_dir/CosyVoice-BlankEN $final_model_dir
  fi
  cp $pretrained_model_dir/campplus.onnx $final_model_dir
  cp $pretrained_model_dir/speech_tokenizer_v2.onnx $final_model_dir
  cp conf/cosyvoice2.yaml $final_model_dir

  # # Save backbone information for inference auto-detection
  # echo "Saving backbone information: $backbone"
  # echo "$backbone" > $final_model_dir/backbone_info.txt


  echo "Exporting optimized models for inference..."
  # if [ -n "$HF_REPO_ID" ]; then
  #   echo "Exporting and uploading to HuggingFace Hub..."
  #   python cosyvoice/bin/export_jit.py --model_dir $final_model_dir --hf_repo_id $HF_REPO_ID --final
  #   python cosyvoice/bin/export_onnx.py --model_dir $final_model_dir --hf_repo_id $HF_REPO_ID --final
  # else
    echo "Exporting locally only..."
    python cosyvoice/bin/export_jit.py --model_dir $final_model_dir --final
    python cosyvoice/bin/export_onnx.py --model_dir $final_model_dir --final
  # fi
  echo "Model export completed: $final_model_dir"
fi


if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  echo "Uploading final model dir to Hugging Face..."

  # upload the final folder to HF
  if [ -n "$HF_REPO_ID" ]; then
    echo "Uploading final folder to HuggingFace Hub..."
    # hf upload Luka512/CosyVoice2-0.5B-EU /tsi/hi-paris/tts/Luka/exp-H100/cosyvoice2/final_model
    hf upload $HF_REPO_ID $final_model_dir
  fi

  echo "Upload completed successfully."

fi
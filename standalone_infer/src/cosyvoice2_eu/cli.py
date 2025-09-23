import argparse
import os
import torchaudio
import torch

from huggingface_hub import snapshot_download

# Import the vendored runtime
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Global model cache for keeping model in memory
_cached_model = None
_cached_model_config = None


def get_or_load_model(model_dir, setting, llm_run_id, flow_run_id, hifigan_run_id, final, backbone):
    """Get cached model or load new one if configuration changed."""
    global _cached_model, _cached_model_config
    
    current_config = {
        'model_dir': model_dir,
        'setting': setting,
        'llm_run_id': llm_run_id,
        'flow_run_id': flow_run_id,
        'hifigan_run_id': hifigan_run_id,
        'final': final,
        'backbone': backbone,
    }
    
    if _cached_model is None or _cached_model_config != current_config:
        print("Loading model...")
        _cached_model = CosyVoice2(
            model_dir,
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False,
            setting=setting,
            llm_run_id=llm_run_id,
            flow_run_id=flow_run_id,
            hifigan_run_id=hifigan_run_id,
            final=final,
            backbone=backbone,
        )
        _cached_model_config = current_config
        print("Model loaded and cached.")
    else:
        print("Using cached model.")
    
    return _cached_model


def main():
    parser = argparse.ArgumentParser(description='CosyVoice2 European Inference (cross-lingual cloning)')
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True, help='Path to a ≥16kHz prompt wav')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default=os.path.expanduser('~/.cache/cosyvoice2-eu'))
    parser.add_argument('--repo-id', type=str, default='Luka512/CosyVoice2-0.5B-EU', help='HF repo to auto-download into --model-dir unless --no-hf is set')
    parser.add_argument('--no-hf', action='store_true', help='Do not download from HF; assume --model-dir already exists')
    parser.add_argument('--setting', type=str, default='llm_flow_hifigan', help='original|llm|flow|hifigan|llm_flow|llm_hifigan|flow_hifigan|llm_flow_hifigan')
    parser.add_argument('--llm-run-id', type=str, default='latest')
    parser.add_argument('--flow-run-id', type=str, default='latest')
    parser.add_argument('--hifigan-run-id', type=str, default='latest')
    parser.add_argument('--final', action='store_true', help='Use final checkpoints (llm.pt/flow.pt/hift.pt)')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--text-frontend', action='store_true', help='Enable text normalization frontend (disabled by default)')
    parser.add_argument('--backbone', type=str, default='blanken', help='LLM backbone (always uses blanken for cosyvoice2-eu)')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cached model and reload from scratch')
    args = parser.parse_args()

    model_dir = args.model_dir
    if not args.no_hf:
        snapshot_download(repo_id=args.repo_id, local_dir=model_dir)

    # Clear cached model if requested
    if args.clear_cache:
        global _cached_model, _cached_model_config
        _cached_model = None
        _cached_model_config = None
        print("Model cache cleared.")

    # Get or load model with caching
    cosyvoice = get_or_load_model(
        model_dir=model_dir,
        setting=args.setting,
        llm_run_id=args.llm_run_id,
        flow_run_id=args.flow_run_id,
        hifigan_run_id=args.hifigan_run_id,
        final=(args.final or not args.no_hf),
        backbone=args.backbone,
    )

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    prompt_speech_16k = load_wav(args.prompt, 16000)
    
    # Collect all generated audio segments
    audio_segments = []
    for i, j in enumerate(
        cosyvoice.inference_cross_lingual(
            args.text,
            prompt_speech_16k,
            stream=args.stream,
            speed=args.speed,
            text_frontend=args.text_frontend,  # Now defaults to False
        )
    ):
        audio_segments.append(j['tts_speech'])
    
    # Concatenate all segments into one final audio
    if len(audio_segments) == 1:
        final_audio = audio_segments[0]
    else:
        # Concatenate along the time dimension (dim=1 for audio tensors)
        final_audio = torch.cat(audio_segments, dim=1)
        print(f"Concatenated {len(audio_segments)} audio segments into final output.")
    
    # Save the final concatenated audio
    torchaudio.save(args.out, final_audio, cosyvoice.sample_rate)



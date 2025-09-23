#!/usr/bin/env python3
"""
CosyVoice2 Model Wrapper for Batch Inference

Integrates your inference logic with batch processing capabilities.
"""

import sys
import os
from pathlib import Path
import torch
import torchaudio
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
import time

# Add CosyVoice paths
sys.path.append('/home/infres/horstmann-24/TTS2/cosy_repo/third_party/Matcha-TTS')
sys.path.append('/home/infres/horstmann-24/TTS2/cosy_repo')

try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
except ImportError as e:
    logging.error(f"Failed to import CosyVoice2: {e}")
    logging.error("Make sure you're running from the correct environment")

logger = logging.getLogger(__name__)


class CosyVoice2Synthesizer:
    """Wrapper for CosyVoice2 with batch processing."""
    
    def __init__(self, model_config: Dict, device: str = "cuda"):
        self.config = model_config
        self.device = device
        self.model = None
        self.prompt_speech = None
        self.cached_spk_id: Optional[str] = None
        
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()
        
    def cleanup(self):
        """Clean up model resources."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        
    def load_model(self):
        """Load the CosyVoice2 model."""
        if self.model is not None:
            return
            
        logger.info(f"Loading CosyVoice2 model: {self.config['setting']}")
        
        try:
            self.model = CosyVoice2(
                model_dir=self.config['model_dir'],
                load_jit=False,
                load_trt=False, 
                load_vllm=self.config.get('use_vllm', False),
                fp16=self.config.get('fp16', False),
                setting=self.config['setting'],
                llm_run_id=self.config.get('llm_run_id'),
                flow_run_id=self.config.get('flow_run_id'),
                hifigan_run_id=self.config.get('hifigan_run_id'),
                final=self.config.get('final', False),
                backbone=self.config.get('backbone'),
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_prompt_audio(self, prompt_path: str):
        """Load prompt audio for voice cloning."""
        if self.prompt_speech is not None:
            return
            
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt audio not found: {prompt_path}")
            
        try:
            self.prompt_speech = load_wav(prompt_path, 16000)
            logger.info(f"Loaded prompt audio: {prompt_path}")
        except Exception as e:
            logger.error(f"Failed to load prompt audio: {e}")
            raise
    
    def _ensure_prompt_cached(self, inference_config: Dict):
        """Optionally cache prompt as a zero-shot speaker for faster reuse."""
        try:
            prompt_text = inference_config.get('prompt_text')
            zero_shot_spk_id = inference_config.get('zero_shot_spk_id')
            if prompt_text and zero_shot_spk_id and self.cached_spk_id != zero_shot_spk_id:
                # Register cached speaker once
                ok = self.model.add_zero_shot_spk(prompt_text, self.prompt_speech, zero_shot_spk_id)
                if ok:
                    self.cached_spk_id = zero_shot_spk_id
                    logger.info(f"Cached zero-shot speaker '{zero_shot_spk_id}' from prompt.")
        except Exception as e:
            logger.warning(f"Failed to cache zero-shot speaker: {e}")
    
    def _concat_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate a list of [1, T] tensors along time."""
        if not outputs:
            raise RuntimeError("No output from model")
        # Ensure CPU tensors
        chunks = [o.cpu() for o in outputs]
        # Some generators may already return full utterance; handle single element fast path
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=1)
    
    def synthesize_single(self, text: str, inference_config: Dict) -> torch.Tensor:
        """Synthesize a single utterance and return a single audio tensor (concatenated over chunks)."""
        if self.model is None:
            self.load_model()
        if self.prompt_speech is None:
            raise ValueError("Prompt speech not loaded")
        
        method = inference_config.get('method', 'cross_lingual')
        speed = inference_config.get('speed', 1.0)
        text_frontend = inference_config.get('text_frontend', False)
        zero_shot_spk_id = inference_config.get('zero_shot_spk_id', '') or ''
        
        # Optionally cache prompt resources for reuse
        self._ensure_prompt_cached(inference_config)
        
        try:
            # Run model and concatenate all yielded chunks
            outputs: List[torch.Tensor] = []
            if method == 'cross_lingual':
                for out in self.model.inference_cross_lingual(
                    text,
                    self.prompt_speech,
                    zero_shot_spk_id=zero_shot_spk_id,
                    stream=False,
                    speed=speed,
                    text_frontend=text_frontend,
                ):
                    outputs.append(out['tts_speech'])
            elif method == 'zero_shot':
                prompt_text = inference_config.get('prompt_text', '')
                for out in self.model.inference_zero_shot(
                    text,
                    prompt_text,
                    self.prompt_speech,
                    zero_shot_spk_id=zero_shot_spk_id,
                    stream=False,
                    speed=speed,
                    text_frontend=text_frontend,
                ):
                    outputs.append(out['tts_speech'])
            elif method == 'instruct2':
                instruct_text = inference_config.get('instruct_text', '')
                for out in self.model.inference_instruct2(
                    text,
                    instruct_text,
                    self.prompt_speech,
                    stream=False,
                    speed=speed,
                    text_frontend=text_frontend,
                ):
                    outputs.append(out['tts_speech'])
            else:
                raise ValueError(f"Unknown inference method: {method}")
            
            return self._concat_outputs(outputs)
        except Exception as e:
            logger.error(f"Synthesis failed for text: {text[:50]}...")
            logger.error(f"Error: {e}")
            raise
    
    def synthesize_batch(self, samples: List[Dict], inference_config: Dict, 
                        output_dir: Optional[str] = None) -> List[Dict]:
        """
        Synthesize a batch of samples.
        
        Args:
            samples: List of sample dicts with 'text' and 'utterance_id' keys
            inference_config: Inference configuration
            output_dir: Directory to save audio files (optional)
            
        Returns:
            List of results with synthesized audio tensors
        """
        if self.model is None:
            self.load_model()
        if self.prompt_speech is None:
            raise ValueError("Prompt speech not loaded")
        
        # Optionally register zero-shot speaker to cache prompt features
        self._ensure_prompt_cached(inference_config)
        
        # Optional warm-up
        if inference_config.get('warmup', False):
            try:
                _ = self.synthesize_single("warmup.", inference_config)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Warm-up failed: {e}")
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Synthesizing {len(samples)} samples...")
        
        # Threaded concurrency
        workers = int(inference_config.get('workers', inference_config.get('batch_size', 1)) or 1)
        timeout_s = float(inference_config.get('timeout_s', 30))
        results: List[Dict] = [None] * len(samples)
        
        def _worker(idx: int, sample: Dict):
            t0 = time.time()
            text = sample['text']
            if inference_config.get('add_language_hint', False) and inference_config.get('language') in ['fr', 'de']:
                hint = "<|fr|><|endofprompt|> " if inference_config.get('language') == 'fr' else "<|de|><|endofprompt|> "
                text = f"{hint}{text}"

            audio_tensor = self.synthesize_single(text, inference_config)
            elapsed = time.time() - t0
            return idx, audio_tensor, elapsed
        
        if workers <= 1:
            for idx, sample in enumerate(samples):
                try:
                    i, audio_tensor, elapsed = _worker(idx, sample)
                    audio_path = None
                    if output_dir:
                        audio_path = Path(output_dir) / f"{sample['utterance_id']}.wav"
                        torchaudio.save(str(audio_path), audio_tensor, self.model.sample_rate)
                    results[i] = {
                        'utterance_id': sample['utterance_id'],
                        'audio_tensor': audio_tensor,
                        'audio_path': str(audio_path) if audio_path else None,
                        'sample_rate': self.model.sample_rate,
                        'synthesis_time': elapsed,
                    }
                except Exception as e:
                    logger.error(f"Failed to synthesize {sample['utterance_id']}: {e}")
                    results[idx] = {
                        'utterance_id': sample['utterance_id'],
                        'audio_tensor': None,
                        'audio_path': None,
                        'sample_rate': None,
                        'synthesis_time': 0.0,
                        'error': str(e),
                    }
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_map = {ex.submit(_worker, idx, sample): idx for idx, sample in enumerate(samples)}
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    sample = samples[idx]
                    try:
                        i, audio_tensor, elapsed = fut.result(timeout=timeout_s)
                        audio_path = None
                        if output_dir:
                            audio_path = Path(output_dir) / f"{sample['utterance_id']}.wav"
                            torchaudio.save(str(audio_path), audio_tensor, self.model.sample_rate)
                        results[i] = {
                            'utterance_id': sample['utterance_id'],
                            'audio_tensor': audio_tensor,
                            'audio_path': str(audio_path) if audio_path else None,
                            'sample_rate': self.model.sample_rate,
                            'synthesis_time': elapsed,
                        }
                    except FuturesTimeout:
                        logger.error(f"Timeout synthesizing {sample['utterance_id']} after {timeout_s}s")
                        results[idx] = {
                            'utterance_id': sample['utterance_id'],
                            'audio_tensor': None,
                            'audio_path': None,
                            'sample_rate': None,
                            'synthesis_time': timeout_s,
                            'error': f'timeout {timeout_s}s',
                        }
                    except Exception as e:
                        logger.error(f"Failed to synthesize {sample['utterance_id']}: {e}")
                        results[idx] = {
                            'utterance_id': sample['utterance_id'],
                            'audio_tensor': None,
                            'audio_path': None,
                            'sample_rate': None,
                            'synthesis_time': 0.0,
                            'error': str(e),
                        }
        
        success_count = sum(1 for r in results if r and r['audio_tensor'] is not None)
        logger.info(f"Synthesis complete: {success_count}/{len(samples)} successful")
        
        return results
    
    def get_sample_rate(self) -> int:
        """Get the model's sample rate."""
        if self.model is None:
            self.load_model()
        return self.model.sample_rate


def test_synthesizer():
    """Test the synthesizer."""
    model_config = {
        'model_dir': '/home/infres/horstmann-24/TTS2/cosy_repo/pretrained_models/CosyVoice2-0.5B',
        'setting': 'llm_flow_hifigan',
        'llm_run_id': 'latest',
        'flow_run_id': 'latest', 
        'hifigan_run_id': 'latest'
    }
    
    inference_config = {
        'method': 'cross_lingual',
        'speed': 1.0,
        'text_frontend': True,
        # Optional improvements
        'workers': 2,
        'timeout_s': 45,
        'warmup': True,
        'prompt_text': 'Bonjour, je suis votre voix de référence.',
        'zero_shot_spk_id': 'eval_cached_prompt',
        'use_vllm': False,
    }
    
    # Test samples
    samples = [
        {
            'utterance_id': 'test_001',
            'text': 'Bonjour, ceci est un test de synthèse vocale.'
        },
        {
            'utterance_id': 'test_002', 
            'text': "Comment allez-vous aujourd'hui?"
        }
    ]
    
    synthesizer = CosyVoice2Synthesizer(model_config)
    prompt_path = '/home/infres/horstmann-24/TTS2/cosy_repo/prompt_audio/macron.wav'
    
    try:
        synthesizer.load_prompt_audio(prompt_path)
        results = synthesizer.synthesize_batch(samples, inference_config, output_dir='test_output')
        
        print(f"Synthesized {len(results)} samples")
        for result in results:
            if result['audio_tensor'] is not None:
                print(f"  {result['utterance_id']}: {result['synthesis_time']:.2f}s")
            else:
                print(f"  {result['utterance_id']}: FAILED - {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_synthesizer()

#!/usr/bin/env python3
"""
Baseline TTS Models Synthesizer

Simple synthesizer for CoquiTTS (XTTS v2) baseline comparison.
"""

import os
import io
import requests
from pathlib import Path
import torch
import torchaudio
import numpy as np
from typing import List, Dict, Optional
import logging
import time
import tempfile

# Optional imports (only needed for OpenVoice baseline)
try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - optional
    sf = None

logger = logging.getLogger(__name__)

#############################################
# CoquiTTS Wrapper
#############################################


class CoquiTTSSynthesizer:
    """Wrapper for CoquiTTS XTTS v2 model."""
    
    def __init__(self, language: str = "fr", device: str = "cuda"):
        self.language = language
        self.device = device
        self.model = None
        self.prompt_audio_path = None
        self._setup_model()
    
    def _setup_model(self):
        """Initialize CoquiTTS XTTS v2 model."""
        try:
            from TTS.api import TTS
        except ImportError:
            logger.error("CoquiTTS not installed. Install with: pip install TTS")
            raise
        
        logger.info("Loading CoquiTTS XTTS v2 model...")
        
        try:
            # Use XTTS v2 multilingual model
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            logger.info("CoquiTTS XTTS v2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CoquiTTS model: {e}")
            raise
    
    def load_prompt_audio(self, prompt_path: str):
        """Load prompt audio for voice cloning."""
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt audio not found: {prompt_path}")
        
        self.prompt_audio_path = prompt_path
        logger.info(f"Loaded prompt audio for CoquiTTS: {prompt_path}")
    
    def synthesize_single(self, text: str) -> np.ndarray:
        """
        Synthesize single utterance with CoquiTTS.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio as numpy array (22kHz, float32)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        if self.prompt_audio_path is None:
            raise ValueError("Prompt audio not loaded")
        
        try:
            # Generate speech
            wav = self.model.tts(
                text=text,
                speaker_wav=self.prompt_audio_path,
                language=self.language
            )
            
            # Convert to numpy array if needed and ensure float32
            if isinstance(wav, list):
                wav = np.array(wav, dtype=np.float32)
            elif isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy().astype(np.float32)
            else:
                wav = wav.astype(np.float32)  # Ensure float32
            
            return wav
            
        except Exception as e:
            logger.error(f"CoquiTTS synthesis failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up model resources."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error during CoquiTTS cleanup: {e}")


class OpenVoiceSynthesizer:
    """Wrapper for OpenVoice tone color conversion baseline using Gemini TTS as base speaker.

    Workflow per text:
      1. Use Gemini TTS to synthesize base speaker audio in target language.
      2. Convert base audio tone color to reference speaker using ToneColorConverter.
    """

    def __init__(self,
                 ckpt_converter: Optional[str],
                 reference_speaker: str,
                 language: str = "fr",
                 device: str = "cuda",
                 gemini_model: str = "gemini-2.5-flash-preview-tts",
                 base_voice: str = "Kore",
                 prompt_prefix_map: Optional[Dict[str, str]] = None,
                 env_file: Optional[str] = None):
        self.ckpt_converter = ckpt_converter.rstrip('/') if ckpt_converter else None
        self.reference_speaker = reference_speaker
        self.language = language
        self.device = device
        self.gemini_model = gemini_model
        self.base_voice = base_voice
        self.prompt_prefix_map = prompt_prefix_map or {
            'fr': 'Speak in French:',
            'de': 'Sprich auf Deutsch:',
            'en': 'Speak in English:'
        }
        self.env_file = env_file
        self.converter = None
        self.source_se = None
        self.target_se = None
        self.client = None
        self._setup()

    def _setup(self):
        try:
            from openvoice.api import ToneColorConverter
            from openvoice import se_extractor
        except ImportError as e:  # pragma: no cover - environment specific
            raise ImportError("OpenVoice not installed. Activate 'openvoice' conda env.") from e

        # Google genai client
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise ImportError("google-genai package not installed in this environment.") from e

        # Load .env if provided
        if self.env_file and os.path.exists(self.env_file):
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(self.env_file)
            except Exception:  # pragma: no cover
                pass
        else:
            # fallback: load default .env in cwd if present
            if os.path.exists('.env'):
                try:
                    from dotenv import load_dotenv  # type: ignore
                    load_dotenv()
                except Exception:
                    pass

        if 'GOOGLE_API_KEY' not in os.environ:
            raise EnvironmentError("GOOGLE_API_KEY not found (ensure .env contains GOOGLE_API_KEY=...).")

        # Allow OPENVOICE_CKPT override via env if ckpt_converter None
        if self.ckpt_converter is None:
            env_ckpt = os.environ.get('OPENVOICE_CKPT')
            if not env_ckpt:
                raise ValueError("OpenVoice checkpoint path not provided (ckpt_converter or OPENVOICE_CKPT env)")
            self.ckpt_converter = env_ckpt.rstrip('/')

        logger.info("Loading OpenVoice ToneColorConverter checkpoint...")
        config_json = f"{self.ckpt_converter}/config.json"
        ckpt_path = f"{self.ckpt_converter}/checkpoint.pth"

        self.converter = ToneColorConverter(config_json, device=self.device)
        self.converter.load_ckpt(ckpt_path)

        # Create API client (stateless usage per request still fine)
        from google import genai
        self.client = genai.Client()

        # Extract tone color embeddings (source: one base utterance; target: reference speaker)
        logger.info("Extracting target speaker embedding (reference)...")
        self.target_se, _ = se_extractor.get_se(self.reference_speaker, self.converter, vad=True)

        # Language-specific long base text (for richer embedding)
        long_base_texts = {
            'fr': "Say in French:\nBonjour et bienvenue ! Aujourd’hui, nous allons découvrir la beauté de la langue française. Écoutez attentivement chaque mot et chaque intonation. La clarté et le rythme sont très importants pour bien comprendre et bien parler. Prenez votre temps, respirez, et profitez de ce moment d’apprentissage.",
            'de': "Sprich auf Deutsch:\nHallo und herzlich willkommen! Heute entdecken wir die Schönheit der deutschen Sprache. Höre genau auf jedes Wort und jede Betonung. Klarheit und Rhythmus sind sehr wichtig, um gut zu verstehen und deutlich zu sprechen. Nimm dir Zeit, atme ruhig und genieße diesen Moment des Lernens.",
        }
        base_text = long_base_texts.get(self.language, self._prompt_prefix() + " Hello.")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_path = tmp_wav.name
        try:
            self._generate_base_speech(base_text, tmp_path)
            self.source_se, _ = se_extractor.get_se(tmp_path, self.converter, vad=True)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        logger.info("OpenVoice baseline ready.")

    def _prompt_prefix(self) -> str:
        return self.prompt_prefix_map.get(self.language, 'Speak:')

    def _generate_base_speech(self, text: str, dst_path: str):
        """Call Gemini TTS to produce base speaker waveform saved to dst_path."""
        import wave
        from google.genai import types  # type: ignore

        response = self.client.models.generate_content(
            model=self.gemini_model,
            contents=f"{text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.base_voice)
                    )
                ),
            ),
        )
        data = response.candidates[0].content.parts[0].inline_data.data
        # Save PCM bytes as wav (Gemini returns linear PCM 24kHz)
        with wave.open(dst_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(data)

    def synthesize_single(self, text: str, output_path: Optional[str] = None) -> np.ndarray:
        """Generate speech for text using Gemini + tone color conversion."""
        if self.converter is None:
            raise RuntimeError("Converter not initialized")
        # Generate base audio for content
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_base:
            base_path = tmp_base.name
        try:
            self._generate_base_speech(self._prompt_prefix() + f"\n{text}", base_path)
            # Run conversion
            conv_out = output_path or (base_path + '.converted.wav')
            self.converter.convert(
                audio_src_path=base_path,
                src_se=self.source_se,
                tgt_se=self.target_se,
                output_path=conv_out,
                message='@OpenVoiceBaseline'
            )
            # Load converted audio
            if sf is None:
                raise ImportError("soundfile not available to load converted audio")
            audio, sr = sf.read(conv_out)
            # Normalize to float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            # Resample to 16k for metrics consistency
            if sr != 16000:
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio_tensor = resampler(audio_tensor).squeeze(0)
                audio = audio_tensor.numpy().astype(np.float32)
            return audio
        finally:
            try:
                os.remove(base_path)
            except OSError:
                pass

    def cleanup(self):  # pragma: no cover - simple resource release
        try:
            del self.converter
        except Exception:
            pass
        self.converter = None


class FishSpeechSynthesizer:
    """Wrapper for Fish Speech TTS API baseline.
    
    Fish Speech runs as a server and provides REST API for synthesis.
    Requires the server to be running on specified host:port.
    """
    
    def __init__(self, 
                 api_url: str = "http://127.0.0.1:8080/v1/tts",
                 reference_audio: Optional[str] = None,
                 language: str = "fr",
                 output_format: str = "wav"):
        self.api_url = api_url
        self.reference_audio = reference_audio
        self.language = language
        self.output_format = output_format
        self.session = requests.Session()
        self._setup()
    
    def _setup(self):
        """Test connection to Fish Speech server."""
        try:
            # Test if server is responding
            test_url = self.api_url.replace('/v1/tts', '/health') if '/v1/tts' in self.api_url else self.api_url
            response = self.session.get(test_url, timeout=5)
            logger.info(f"Fish Speech server connection test: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not test Fish Speech server connection: {e}")
    
    def load_reference_audio(self, reference_path: str):
        """Set reference audio for voice cloning."""
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference audio not found: {reference_path}")
        self.reference_audio = reference_path
        logger.info(f"Loaded reference audio for Fish Speech: {reference_path}")
    
    def synthesize_single(self, text: str) -> np.ndarray:
        """
        Synthesize single utterance with Fish Speech API.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio as numpy array (16kHz, float32)
        """
        if self.reference_audio is None:
            raise ValueError("Reference audio not loaded")
        
        try:
            # Import required modules for Fish Speech API format
            try:
                import ormsgpack
            except ImportError:
                raise ImportError("ormsgpack required for Fish Speech API. Install with: pip install ormsgpack")
            
            # Read reference audio as binary data
            with open(self.reference_audio, 'rb') as f:
                audio_bytes = f.read()
            
            # Prepare data in Fish Speech API format
            # Based on api_client.py, we need ServeReferenceAudio and ServeTTSRequest format
            reference_audio_data = {
                "audio": audio_bytes,
                "text": ""  # Empty text for reference audio
            }
            
            request_data = {
                "text": text,
                "references": [reference_audio_data],
                "reference_id": None,
                "format": self.output_format,
                "max_new_tokens": 1024,
                "chunk_length": 300,
                "top_p": 0.8,
                "repetition_penalty": 1.1,
                "temperature": 0.8,
                "streaming": False,
                "use_memory_cache": "off",
                "seed": None
            }
            
            # Serialize with MessagePack
            packed_data = ormsgpack.packb(request_data)
            
            # Make API request with correct headers
            headers = {
                "content-type": "application/msgpack",
                "authorization": "Bearer YOUR_API_KEY"  # Fish Speech might ignore this
            }
            
            response = self.session.post(
                self.api_url,
                data=packed_data,
                headers=headers,
                timeout=60  # 60 second timeout for synthesis
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Fish Speech API error {response.status_code}: {response.text}")
            
            # Parse audio response
            audio_bytes = response.content
            
            # Load audio using torchaudio or soundfile
            try:
                # Try with temporary file approach
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                
                # Load with torchaudio
                audio_tensor, sample_rate = torchaudio.load(tmp_path)
                os.unlink(tmp_path)  # Clean up temp file
                
                # Convert to mono if stereo
                if audio_tensor.shape[0] > 1:
                    audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    audio_tensor = resampler(audio_tensor)
                
                # Convert to numpy array
                audio_array = audio_tensor.squeeze().numpy().astype(np.float32)
                
                return audio_array
                
            except Exception as e:
                logger.error(f"Failed to parse audio response: {e}")
                raise
            
        except Exception as e:
            logger.error(f"Fish Speech synthesis failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.session.close()
        except Exception as e:
            logger.warning(f"Error during Fish Speech cleanup: {e}")


class BaselinesSynthesizer:
    """Unified interface for multiple baseline TTS models (CoquiTTS, OpenVoice)."""

    def __init__(self, language: str = "fr", device: str = "cuda"):
        self.language = language
        self.device = device
        self.coqui = None  # CoquiTTSSynthesizer
        self.openvoice = None  # OpenVoiceSynthesizer
        self.elevenlabs = None  # ElevenLabsSynthesizer instance when loaded
        self.fishspeech = None  # FishSpeechSynthesizer instance when loaded

    # -------- Common helpers --------
    def _get_prompt_audio_for_language(self, prompt_config: Dict) -> str:
        if isinstance(prompt_config, dict):
            if self.language and self.language in prompt_config:
                return prompt_config[self.language]
            return prompt_config.get('default', next(iter(prompt_config.values())))
        return prompt_config

    # -------- CoquiTTS --------
    def load_coqui_model(self):
        if self.coqui is None:
            logger.info("Loading CoquiTTS baseline model...")
            self.coqui = CoquiTTSSynthesizer(self.language, self.device)

    def load_coqui_prompt_audio(self, prompt_config: Dict):
        prompt_path = self._get_prompt_audio_for_language(prompt_config)
        if self.coqui is not None:
            self.coqui.load_prompt_audio(prompt_path)

    # -------- OpenVoice --------
    def load_openvoice_model(self, ov_config: Dict, prompt_config: Dict):
        if self.openvoice is not None:
            return
        ckpt = ov_config.get('ckpt_converter') or ov_config.get('OPENVOICE_CKPT')
        # ckpt may be None -> will fallback to env OPENVOICE_CKPT inside OpenVoiceSynthesizer
        reference_speaker = ov_config.get('reference_speaker') or self._get_prompt_audio_for_language(prompt_config)
        gemini_model = ov_config.get('gemini_model', 'gemini-2.5-flash-preview-tts')
        base_voice = ov_config.get('base_voice', 'Kore')
        prompt_prefix_map = ov_config.get('prompt_prefix_map')
        env_file = ov_config.get('env_file')  # optional path to .env
        logger.info("Loading OpenVoice baseline model...")
        self.openvoice = OpenVoiceSynthesizer(
            ckpt_converter=ckpt,
            reference_speaker=reference_speaker,
            language=self.language,
            device=self.device,
            gemini_model=gemini_model,
            base_voice=base_voice,
            prompt_prefix_map=prompt_prefix_map,
            env_file=env_file
        )

    # -------- Unified synthesis --------
    def synthesize_batch(self,
                         samples: List[Dict],
                         output_dir: Optional[str] = None,
                         model: str = 'coqui') -> List[Dict]:
        if model == 'coqui':
            if self.coqui is None:
                raise ValueError("CoquiTTS model not loaded")
            logger.info(f"Synthesizing {len(samples)} samples with CoquiTTS baseline...")
            synth_fn = self.coqui.synthesize_single
            src_sample_rate = 22050
        elif model == 'openvoice':
            if self.openvoice is None:
                raise ValueError("OpenVoice model not loaded")
            logger.info(f"Synthesizing {len(samples)} samples with OpenVoice baseline (Gemini + conversion)...")
            synth_fn = self.openvoice.synthesize_single
            src_sample_rate = 16000  # we resample inside already; treat as final
        elif model == 'elevenlabs':
            if self.elevenlabs is None:
                raise ValueError("ElevenLabs model not loaded")
            logger.info(f"Synthesizing {len(samples)} samples with ElevenLabs baseline...")
            synth_fn = self.elevenlabs.synthesize_single
            src_sample_rate = 16000
        elif model == 'fishspeech':
            if self.fishspeech is None:
                raise ValueError("Fish Speech model not loaded")
            logger.info(f"Synthesizing {len(samples)} samples with Fish Speech baseline...")
            synth_fn = self.fishspeech.synthesize_single
            src_sample_rate = 16000
        else:
            raise ValueError(f"Unknown baseline model: {model}")

        results: List[Dict] = []
        path_obj = None
        if output_dir:
            path_obj = Path(output_dir)
            path_obj.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(samples):
            try:
                start = time.time()
                audio_array = synth_fn(sample['text'])
                synth_time = time.time() - start
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                # Coqui resample step
                if model == 'coqui' and src_sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(src_sample_rate, 16000)
                    audio_tensor = resampler(audio_tensor)
                audio_tensor = audio_tensor.float()
                audio_path = None
                if path_obj:
                    audio_path = path_obj / f"{sample['utterance_id']}.wav"
                    torchaudio.save(str(audio_path), audio_tensor, 16000)
                results.append({
                    'utterance_id': sample['utterance_id'],
                    'audio_tensor': audio_tensor,
                    'audio_path': str(audio_path) if audio_path else None,
                    'sample_rate': 16000,
                    'synthesis_time': synth_time,
                    'error': None
                })
                if (i + 1) % 10 == 0:
                    logger.info(f"{model}: processed {i + 1}/{len(samples)} samples")
            except Exception as e:  # pragma: no cover - runtime errors
                logger.error(f"{model} synthesis failed for {sample['utterance_id']}: {e}")
                results.append({
                    'utterance_id': sample['utterance_id'],
                    'audio_tensor': None,
                    'audio_path': None,
                    'sample_rate': None,
                    'synthesis_time': 0.0,
                    'error': str(e)
                })

        ok = len([r for r in results if r['audio_tensor'] is not None])
        logger.info(f"{model} baseline completed: {ok}/{len(results)} successful")
        return results

    def cleanup(self):
        if self.coqui is not None:
            self.coqui.cleanup()
            self.coqui = None
        if self.openvoice is not None:
            self.openvoice.cleanup()
            self.openvoice = None
        if self.elevenlabs is not None:
            self.elevenlabs.cleanup()
            self.elevenlabs = None
        if self.fishspeech is not None:
            self.fishspeech.cleanup()
            self.fishspeech = None

    # -------- ElevenLabs --------
    def load_elevenlabs_model(self, el_config: Dict, prompt_config: Dict):
        if self.elevenlabs is not None:
            return
        logger.info("Loading ElevenLabs baseline model...")
        api_key = el_config.get('api_key')  # strongly prefer env var; direct key optional
        env_file = el_config.get('env_file')
        model_id = el_config.get('model_id', 'eleven_turbo_v2')
        voice_map = el_config.get('voice_ids', {})
        # fallback: use prompt audio mapping language key to choose voice id
        voice_id = voice_map.get(self.language) or voice_map.get('default')
        if not voice_id:
            raise ValueError("ElevenLabs voice id not provided for language or default.")
        self.elevenlabs = ElevenLabsSynthesizer(
            api_key=api_key,
            env_file=env_file,
            voice_id=voice_id,
            language=self.language,
            model_id=model_id,
        )

    # -------- Fish Speech --------
    def load_fishspeech_model(self, fs_config: Dict, prompt_config: Dict):
        if self.fishspeech is not None:
            return
        logger.info("Loading Fish Speech baseline model...")
        api_url = fs_config.get('api_url', 'http://127.0.0.1:8080/v1/tts')
        output_format = fs_config.get('output_format', 'wav')
        reference_audio = self._get_prompt_audio_for_language(prompt_config)
        
        self.fishspeech = FishSpeechSynthesizer(
            api_url=api_url,
            reference_audio=reference_audio,
            language=self.language,
            output_format=output_format
        )


class ElevenLabsSynthesizer:
    """Simple ElevenLabs REST API wrapper for baseline synthesis.

    Uses streaming /v1/text-to-speech endpoint. Keeps things minimal: we don't cache
    per-text results; focus on low-cost model (default: eleven_turbo_v2). Assumes
    user supplies voice IDs (e.g., cloned Macron / Merz voices) via config.
    """

    def __init__(self,
                 api_key: Optional[str],
                 voice_id: str,
                 language: str = 'fr',
                 model_id: str = 'eleven_turbo_v2',
                 env_file: Optional[str] = None):
        self.api_key = api_key
        self.voice_id = voice_id
        self.language = language
        self.model_id = model_id
        self.env_file = env_file
        self.session = requests.Session()
        self._setup()

    def _setup(self):
        # Load env file if provided
        if self.env_file and os.path.exists(self.env_file):
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(self.env_file)
            except Exception:  # pragma: no cover
                pass
        if not self.api_key:
            self.api_key = os.environ.get('ELEVEN_API_KEY') or os.environ.get('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise EnvironmentError("ElevenLabs API key not provided (set api_key or ELEVEN_API_KEY env)")

    def synthesize_single(self, text: str) -> np.ndarray:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            'xi-api-key': self.api_key,
        }
        payload = {
            'text': text,
            'model_id': self.model_id,
        }
        try:
            r = self.session.post(url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"ElevenLabs request failed: {e}")

        # ElevenLabs returns audio/mpeg by default; request wav? Minimal approach: decode via torchaudio.
        audio_bytes = io.BytesIO(r.content)
        try:
            audio_tensor, sr = torchaudio.load(audio_bytes)
        except Exception as e:
            raise RuntimeError(f"Failed to decode ElevenLabs audio: {e}")
        # Convert to mono float32 16k
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_tensor = resampler(audio_tensor)
        return audio_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    def cleanup(self):  # pragma: no cover - simple
        try:
            self.session.close()
        except Exception:
            pass



def test_baselines_synthesizer():
    """Test function for baseline synthesizer."""
    import tempfile
    
    # Create test samples
    test_samples = [
        {'utterance_id': 'test_1', 'text': 'Bonjour, comment allez-vous?'},
        {'utterance_id': 'test_2', 'text': 'Ceci est un test de synthèse vocale.'}
    ]
    
    # Test prompt audio path
    test_prompt = "/home/infres/horstmann-24/TTS2/cosy_repo/prompt_audio/macron.wav"
    
    if not os.path.exists(test_prompt):
        print(f"Prompt audio not found: {test_prompt}")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        synthesizer = BaselinesSynthesizer(language="fr")
        try:
            synthesizer.load_coqui_model()
            synthesizer.load_coqui_prompt_audio({"fr": test_prompt})
            results = synthesizer.synthesize_batch(test_samples, output_dir=temp_dir, model='coqui')
            print(f"CoquiTTS test results: {len(results)} samples processed")
            successful = len([r for r in results if r['audio_tensor'] is not None])
            print(f"Success rate: {successful}/{len(results)}")
        finally:
            synthesizer.cleanup()

    # Optional OpenVoice quick smoke test if environment vars & checkpoints are present
    ov_ckpt = os.environ.get('OPENVOICE_CKPT')
    if ov_ckpt and os.path.exists(ov_ckpt + '/config.json') and os.environ.get('GOOGLE_API_KEY'):
        print("\nRunning optional OpenVoice baseline smoke test...")
        with tempfile.TemporaryDirectory() as temp_dir:
            ov = BaselinesSynthesizer(language='fr')
            try:
                ov.load_openvoice_model({'ckpt_converter': ov_ckpt, 'reference_speaker': test_prompt}, {"fr": test_prompt})
                res = ov.synthesize_batch(test_samples[:1], output_dir=temp_dir, model='openvoice')
                print(f"OpenVoice generated {len(res)} sample(s)")
            except Exception as e:
                print(f"OpenVoice test skipped/failed: {e}")
            finally:
                ov.cleanup()

    # Optional ElevenLabs smoke test if API key and placeholder voice id provided
    el_api = os.environ.get('ELEVEN_API_KEY') or os.environ.get('ELEVENLABS_API_KEY')
    el_voice = os.environ.get('ELEVEN_VOICE_ID')  # user can export a test voice id
    if el_api and el_voice:
        print("\nRunning optional ElevenLabs baseline smoke test...")
        with tempfile.TemporaryDirectory() as temp_dir:
            el = BaselinesSynthesizer(language='fr')
            try:
                el.load_elevenlabs_model({'api_key': el_api, 'voice_ids': {'fr': el_voice}}, {"fr": test_prompt})
                res = el.synthesize_batch(test_samples[:1], output_dir=temp_dir, model='elevenlabs')
                print(f"ElevenLabs generated {len(res)} sample(s)")
            except Exception as e:
                print(f"ElevenLabs test skipped/failed: {e}")
            finally:
                el.cleanup()

    # Optional Fish Speech smoke test if server is running
    fs_api_url = os.environ.get('FISHSPEECH_API_URL', 'http://127.0.0.1:8080/v1/tts')
    try:
        # Test if Fish Speech server is reachable
        import requests
        response = requests.get(fs_api_url.replace('/v1/tts', '/health'), timeout=2)
        if response.status_code == 200:
            print("Fish Speech server detected, testing...")
            fs = BaselinesSynthesizer(language='fr')
            try:
                fs.load_fishspeech_model({'api_url': fs_api_url}, {"fr": test_prompt})
                res = fs.synthesize_batch(test_samples[:1], output_dir=temp_dir, model='fishspeech')
                print(f"Fish Speech generated {len(res)} sample(s)")
            except Exception as e:
                print(f"Fish Speech test failed: {e}")
            finally:
                fs.cleanup()
        else:
            print("Fish Speech server not available for testing")
    except Exception as e:
        print(f"Fish Speech test skipped: {e}")


if __name__ == "__main__":
    test_baselines_synthesizer()

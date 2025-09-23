#!/usr/bin/env python3
"""
Metrics Computing for CosyVoice2 Evaluation (fixed/extended)

Adds:
- WER (raw) and WER_norm (language-aware normalization incl. diacritics/ligatures, punctuation/number normalization)
- Correct MCD in dB using mel-cepstral coefficients (c1..c12, 25ms/10ms, DTW) via WORLD + pysptk
  * If unavailable, falls back to LSD on mel-spectrograms in dB (lsd_mel_db) and sets MCD to NaN.
- SECS with silence trimming
- Pitch metrics: GPE (voiced), F0 RMSE (Hz), F0 correlation, V/UV error
"""

import os
import re
import math
import yaml
import json
import librosa
import numpy as np
import torch
import tempfile
import soundfile as sf
import logging
import unicodedata
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------- Optional deps ----------
try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except Exception:
    FASTDTW_AVAILABLE = False

try:
    import whisper as whisper_pkg
    OPENAI_WHISPER_AVAILABLE = True
except Exception:
    OPENAI_WHISPER_AVAILABLE = False

try:
    from jiwer import wer as compute_jiwer_wer
    JIWER_AVAILABLE = True
except Exception:
    JIWER_AVAILABLE = False

# jiwer.cer may not exist in older jiwer; fall back to simple Levenshtein
try:
    from jiwer import cer as compute_jiwer_cer
    JIWER_CER_AVAILABLE = True
except Exception:
    JIWER_CER_AVAILABLE = False

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

# True MCD deps (optional)
try:
    import pyworld
    PYWORLD_AVAILABLE = True
except Exception:
    PYWORLD_AVAILABLE = False

try:
    import pysptk
    PYSPTK_AVAILABLE = True
except Exception:
    PYSPTK_AVAILABLE = False


# ----------------- helpers -----------------
def _levenshtein(a: str, b: str) -> int:
    """Simple Levenshtein distance for CER fallback."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(a) + 1)
    dp[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[-1, -1])


def _nfkc_lower(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower().strip()


def _strip_diacritics(s: str) -> str:
    # Remove combining marks (accents). Keeps base letters.
    return "".join(ch for ch in unicodedata.normalize("NFD", s)
                   if unicodedata.category(ch) != "Mn")


def _normalize_for_wer(s: str, lang: Optional[str], strip_accents: bool = True) -> str:
    """
    Language-aware ASR-style normalization:
      - NFKC + lowercase
      - unify quotes/dashes; map ß→ss (DE), œ/æ→oe/ae (FR)
      - remove thousands separators (., NBSP, NNBSP, thin/hair space), decimal comma -> dot
      - treat hyphens/apostrophes as separators
      - optional diacritics stripping (default True)
      - drop remaining punctuation; collapse whitespace
    """
    s = _nfkc_lower(s)

    # unify quotes/apostrophes/dashes
    s = (s.replace("’", "'").replace("‘", "'").replace("‚", "'")
           .replace("“", '"').replace("”", '"').replace("„", '"')
           .replace("–", "-").replace("—", "-").replace("−", "-"))

    # language-specific character maps
    if lang:
        lang_l = lang.lower()
        if lang_l.startswith("de"):
            s = s.replace("ß", "ss")
        if lang_l.startswith(("fr", "fra", "fre")):
            s = s.replace("œ", "oe").replace("æ", "ae")

    # numbers: remove thousands sep (., NBSP \u00A0, NNBSP \u202F, thin \u2009, hair \u200A, spaces)
    s = re.sub(r'(?<=\d)[\.\u00A0\u202F\u2009\u200A\s](?=\d{3}\b)', '', s)
    # decimal comma -> dot
    s = re.sub(r'(\d),(\d)', r'\1.\2', s)

    # treat hyphen/apostrophe as separators
    s = re.sub(r"[-']", " ", s)

    # optional diacritics strip
    if strip_accents:
        s = _strip_diacritics(s)

    # remove remaining punctuation (keep letters/digits/underscore/space)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _trim_silence(audio: np.ndarray, sr: int, top_db: int = 40) -> np.ndarray:
    """Trim leading/trailing silence; return original if trim fails."""
    try:
        y, _ = librosa.effects.trim(audio, top_db=top_db)
        if y.size > 0:
            return y
    except Exception:
        pass
    return audio


def _whisper_name_cleanup(name: str) -> str:
    """
    Normalize whisper model names; accept 'openai/whisper-large-v3' or 'large-v3'.
    """
    if not name:
        return "large-v3"
    name = name.strip()
    if name.startswith("openai/whisper-"):
        return name.split("openai/whisper-")[-1]
    return name


def _stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def _alpha_for_sr(sr: int) -> float:
    # All-pass constant for mel-cepstra (SPTK convention)
    # Common choices:
    # 16 kHz -> 0.58; 22.05 kHz -> 0.65; 44.1/48 kHz -> 0.76
    if sr <= 16000:
        return 0.58
    if sr <= 22050:
        return 0.65
    return 0.76


class MetricsComputer:
    """Compute acoustic and intelligibility metrics (fixed/extended)."""

    def __init__(self, model_dir: Optional[str] = None, config_path: Optional[str] = None, language: Optional[str] = None):
        self.campplus_session = None
        self.model_dir = model_dir
        self.config = self._load_config(config_path)

        # whisper language hint
        self.language = (language or (self.config or {}).get("language", None))
        if isinstance(self.language, str):
            self.language = self.language.lower()

        # HF cache location
        hf_home = self.config.get("system", {}).get("HF_HOME", "/tsi/hi-paris/tts/Luka/models/")
        os.environ["HF_HOME"] = hf_home
        self.hf_cache_dir = hf_home

        # whisper name
        cfg_name = (self.config.get("metrics", {}).get("whisper_model") or "large-v3")
        self.whisper_name = _whisper_name_cleanup(cfg_name)

        self.whisper_pkg_model = None
        self.whisper_cache_dir = os.path.join(hf_home, "openai-whisper-cache")
        os.makedirs(self.whisper_cache_dir, exist_ok=True)

    # ---------- Config ----------
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "eval_config.yaml")
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}

    # ---------- Whisper loader ----------
    def _load_whisper_pkg(self):
        if not OPENAI_WHISPER_AVAILABLE:
            raise ImportError("openai-whisper not available")
        if self.whisper_pkg_model is not None:
            return
        name = _whisper_name_cleanup(self.whisper_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_pkg_model = whisper_pkg.load_model(name, device=device, download_root=self.whisper_cache_dir)
        logger.info(f"openai-whisper '{name}' loaded on {device}")

    # ---------- MCD (true, dB) via WORLD + pysptk ----------
    def _extract_mcep_sequence(self, y: np.ndarray, sr: int, order: int = 12) -> Optional[np.ndarray]:
        """
        Returns mel-cepstrum sequence of shape (T, order+1) using WORLD (spectral envelope) + pysptk.sp2mc.
        c0..c_order; we will exclude c0 later for MCD.
        """
        if not (PYWORLD_AVAILABLE and PYSPTK_AVAILABLE):
            return None
        try:
            # WORLD analysis
            _y = y.astype(np.float64)
            f0, t = pyworld.dio(_y, sr)
            f0 = pyworld.stonemask(_y, f0, t, sr)
            sp = pyworld.cheaptrick(_y, f0, t, sr)  # spectral envelope (linear amplitude)
            # Convert to mel-cepstra
            alpha = _alpha_for_sr(sr)
            mc = pysptk.sp2mc(sp, order=order, alpha=alpha)  # (T, order+1)
            return mc
        except Exception as e:
            logger.warning(f"MCEP extraction failed: {e}")
            return None

    def compute_mcd(self, reference_path: str, synthesized_path: str, order: int = 12) -> float:
        """
        True MCD [dB] using WORLD spectral envelope + mel-cepstra (c1..c12), 25ms/10ms alignment via DTW.
        If MCEP can't be computed, returns NaN (see lsd fallback).
        """
        try:
            ref, sr_ref = librosa.load(reference_path, sr=None)
            syn, sr_syn = librosa.load(synthesized_path, sr=None)

            # resample both to common SR for consistent WORLD params (16k recommended)
            target_sr = 16000
            if sr_ref != target_sr:
                ref = librosa.resample(ref, orig_sr=sr_ref, target_sr=target_sr)
                sr_ref = target_sr
            if sr_syn != target_sr:
                syn = librosa.resample(syn, orig_sr=sr_syn, target_sr=target_sr)
                sr_syn = target_sr

            ref_mc = self._extract_mcep_sequence(ref, sr_ref, order=order)
            syn_mc = self._extract_mcep_sequence(syn, sr_syn, order=order)
            if ref_mc is None or syn_mc is None:
                return float("nan")

            # Exclude c0 -> use c1..c_order
            ref_ceps = ref_mc[:, 1:order+1].T  # (order, T)
            syn_ceps = syn_mc[:, 1:order+1].T

            # DTW alignment on Euclidean distance
            try:
                D, wp = librosa.sequence.dtw(ref_ceps, syn_ceps, metric="euclidean")
                wp = np.array(wp)[::-1]  # path from start to end
                diffs = ref_ceps[:, wp[:, 0]].T - syn_ceps[:, wp[:, 1]].T  # (L, order)
                frame_dist = np.linalg.norm(diffs, axis=1)  # (L,)
            except Exception:
                if FASTDTW_AVAILABLE:
                    _, path = fastdtw(ref_ceps.T, syn_ceps.T, dist=lambda a, b: np.linalg.norm(a - b))
                    aligned_ref = np.array([ref_ceps.T[i] for i, j in path])
                    aligned_syn = np.array([syn_ceps.T[j] for i, j in path])
                    frame_dist = np.linalg.norm(aligned_ref - aligned_syn, axis=1)
                else:
                    L = min(ref_ceps.shape[1], syn_ceps.shape[1])
                    frame_dist = np.linalg.norm(ref_ceps[:, :L].T - syn_ceps[:, :L].T, axis=1)

            # MCD [dB] = (10/ln(10)) * sqrt(2) * mean(||Δc||)
            const = (10.0 / math.log(10.0)) * math.sqrt(2.0)
            mcd_db = const * float(np.mean(frame_dist))
            return float(mcd_db)
        except Exception as e:
            logger.warning(f"Failed to compute MCD: {e}")
            return float("nan")

    # ---------- LSD (mel dB) fallback ----------
    def compute_lsd_mel_db(self, reference_path: str, synthesized_path: str, n_mels: int = 80) -> float:
        """
        Mel Log-Spectral Distance (mean across aligned frames):
            For aligned frames i, j:
              lsd_frame = sqrt( mean_k (S_ref_dB[k,i] - S_syn_dB[k,j])^2 )
            LSD = mean over path.
        """
        try:
            ref, sr_ref = librosa.load(reference_path, sr=None)
            syn, sr_syn = librosa.load(synthesized_path, sr=None)

            # Common SR and STFT params (25ms win, 10ms hop)
            sr = sr_ref if sr_ref == sr_syn else min(sr_ref, sr_syn)
            if sr_ref != sr:
                ref = librosa.resample(ref, orig_sr=sr_ref, target_sr=sr)
            if sr_syn != sr:
                syn = librosa.resample(syn, orig_sr=sr_syn, target_sr=sr)

            win = int(0.025 * sr)
            hop = int(0.010 * sr)
            n_fft = 2048 if win < 2048 else win

            ref_mel = librosa.feature.melspectrogram(y=ref, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
                                                     n_mels=n_mels, power=2.0)
            syn_mel = librosa.feature.melspectrogram(y=syn, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
                                                     n_mels=n_mels, power=2.0)
            ref_db = librosa.power_to_db(ref_mel, ref=np.max)
            syn_db = librosa.power_to_db(syn_mel, ref=np.max)

            # Align on Euclidean across mel bins
            try:
                D, wp = librosa.sequence.dtw(ref_db, syn_db, metric="euclidean")
                wp = np.array(wp)[::-1]
                diffs = ref_db[:, wp[:, 0]].T - syn_db[:, wp[:, 1]].T  # (L, n_mels)
            except Exception:
                if FASTDTW_AVAILABLE:
                    _, path = fastdtw(ref_db.T, syn_db.T, dist=lambda a, b: np.linalg.norm(a - b))
                    aligned_ref = np.array([ref_db.T[i] for i, j in path])
                    aligned_syn = np.array([syn_db.T[j] for i, j in path])
                    diffs = aligned_ref - aligned_syn
                else:
                    L = min(ref_db.shape[1], syn_db.shape[1])
                    diffs = ref_db[:, :L].T - syn_db[:, :L].T

            lsd_per_frame = np.sqrt(np.mean(diffs ** 2, axis=1))  # (L,)
            return float(np.mean(lsd_per_frame))
        except Exception as e:
            logger.warning(f"Failed to compute LSD mel dB: {e}")
            return float("nan")

    # ---------- WER / CER ----------
    def compute_wer(self, ref_text: str, syn_audio_path: str) -> float:
        """
        Raw WER (%) exactly like before (no normalization).
        """
        if not os.path.exists(syn_audio_path) or not JIWER_AVAILABLE or not OPENAI_WHISPER_AVAILABLE:
            return float("nan")
        try:
            self._load_whisper_pkg()
            audio, sr = sf.read(syn_audio_path, always_2d=False)
            audio = _stereo_to_mono(audio)
            if sr != 16000:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
            elif audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            lang = (self.language if isinstance(self.language, str) and self.language not in ("", "auto") else None)
            result = self.whisper_pkg_model.transcribe(
                audio=audio, language=lang, task="transcribe",
                fp16=torch.cuda.is_available(), temperature=0.0,
                without_timestamps=True, verbose=False
            )
            hyp = (result.get("text") or "").strip()
            ref = (ref_text or "").strip()
            return float(compute_jiwer_wer(ref, hyp) * 100.0)
        except Exception as e:
            logger.warning(f"openai-whisper WER failed: {e}")
            return float("nan")

    def compute_wer_and_norm_with_transcript(self, ref_text: str, syn_audio_path: str):
        """
        Returns dict with raw & normalized WER + CER and transcripts:
        {
          'wer': float or NaN,
          'hyp': str,
          'wer_norm': float or NaN,
          'ref_norm': str,
          'hyp_norm': str,
          'cer': float or NaN,
          'cer_norm': float or NaN
        }
        """
        out = {
            "wer": float("nan"),
            "hyp": "",
            "wer_norm": float("nan"),
            "ref_norm": "",
            "hyp_norm": "",
            "cer": float("nan"),
            "cer_norm": float("nan"),
        }
        if not os.path.exists(syn_audio_path) or not OPENAI_WHISPER_AVAILABLE:
            return out
        try:
            self._load_whisper_pkg()
            audio, sr = sf.read(syn_audio_path, always_2d=False)
            audio = _stereo_to_mono(audio)
            if sr != 16000:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
            elif audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            lang = (self.language if isinstance(self.language, str) and self.language not in ("", "auto") else None)
            result = self.whisper_pkg_model.transcribe(
                audio=audio, language=lang, task="transcribe",
                fp16=torch.cuda.is_available(), temperature=0.0,
                without_timestamps=True, verbose=False
            )
            hyp = (result.get("text") or "").strip()
            ref = (ref_text or "").strip()
            out["hyp"] = hyp

            if JIWER_AVAILABLE:
                try:
                    out["wer"] = float(compute_jiwer_wer(ref, hyp) * 100.0)
                except Exception:
                    pass

            # Normalized strings (now includes diacritics removal and better number handling)
            ref_norm = _normalize_for_wer(ref, self.language, strip_accents=True)
            hyp_norm = _normalize_for_wer(hyp, self.language, strip_accents=True)
            out["ref_norm"] = ref_norm
            out["hyp_norm"] = hyp_norm

            # Normalized WER
            if JIWER_AVAILABLE and ref_norm and hyp_norm:
                out["wer_norm"] = float(compute_jiwer_wer(ref_norm, hyp_norm) * 100.0)

            # CER (raw)
            if ref and hyp:
                if JIWER_CER_AVAILABLE:
                    out["cer"] = float(compute_jiwer_cer(ref, hyp) * 100.0)
                else:
                    d = _levenshtein(ref, hyp)
                    out["cer"] = float((d / max(1, len(ref))) * 100.0)

            # CER (normalized)
            if ref_norm and hyp_norm:
                if JIWER_CER_AVAILABLE:
                    out["cer_norm"] = float(compute_jiwer_cer(ref_norm, hyp_norm) * 100.0)
                else:
                    d = _levenshtein(ref_norm, hyp_norm)
                    out["cer_norm"] = float((d / max(1, len(ref_norm))) * 100.0)

            return out
        except Exception as e:
            logger.warning(f"openai-whisper (norm) failed: {e}")
            return out

    # ---------- Speaker similarity ----------
    def _load_campplus(self):
        if not ONNX_AVAILABLE:
            return
        if self.campplus_session is not None:
            return
        campplus_path = self.config.get("campplus", {}).get("model_path")
        if not campplus_path and self.model_dir:
            campplus_path = f"{self.model_dir}/campplus.onnx"
        if not campplus_path or not os.path.exists(campplus_path):
            logger.debug(f"CamPLUS model not found at: {campplus_path}")
            return
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        try:
            self.campplus_session = onnxruntime.InferenceSession(
                campplus_path, sess_options=option, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            logger.warning(f"Failed to load CamPLUS model: {e}")
            self.campplus_session = None

    def _extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = _trim_silence(audio, sr, top_db=40)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        if sr != 16000:
            import torchaudio.transforms as T
            audio_tensor = T.Resample(orig_freq=sr, new_freq=16000)(audio_tensor)

        import torchaudio.compliance.kaldi as kaldi
        feat = kaldi.fbank(audio_tensor, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)

        emb = self.campplus_session.run(
            None, {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(0).numpy()}
        )[0].flatten()
        return emb

    def compute_secs(self, reference_path: str, synthesized_path: str) -> float:
        """Speaker Embedding Cosine Similarity (with silence trimming)."""
        try:
            self._load_campplus()
            if self.campplus_session is None:
                return float("nan")

            ref_audio, sr_ref = librosa.load(reference_path, sr=None)
            syn_audio, sr_syn = librosa.load(synthesized_path, sr=None)
            ref_emb = self._extract_speaker_embedding(ref_audio, sr_ref)
            syn_emb = self._extract_speaker_embedding(syn_audio, sr_syn)

            dot = float(np.dot(ref_emb, syn_emb))
            nr = float(np.linalg.norm(ref_emb))
            ns = float(np.linalg.norm(syn_emb))
            if nr == 0.0 or ns == 0.0:
                return float("nan")
            return dot / (nr * ns)
        except Exception as e:
            logger.warning(f"SECS computation failed: {e}")
            return float("nan")

    # ---------- Pitch metrics ----------
    def _extract_f0(self, y: np.ndarray, sr: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (f0_hz, vuv_mask) with hop length 'hop'.
        Use librosa.pyin; fall back to yin if pyin fails.
        """
        try:
            f0, vflag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr, frame_length=2048, hop_length=hop)
            vuv = (vflag == True) & (~np.isnan(f0))
            f0 = np.where(vuv, f0, np.nan)
            return f0, vuv
        except Exception:
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr, frame_length=2048, hop_length=hop)
            vuv = ~np.isnan(f0)
            f0 = np.where(vuv, f0, np.nan)
            return f0, vuv

    def compute_pitch_metrics(self, reference_path: str, synthesized_path: str,
                          min_voiced_ms: int = 300, min_coverage_ratio: float = 0.08) -> Dict[str, float]:
        """
        DTW-aligned F0 metrics with minimum voiced-content guard:
        - gpe: Gross Pitch Error % on aligned voiced pairs (|Δf0|/f0_ref > 20%)
        - f0_rmse_hz: RMSE on aligned voiced pairs (Hz)
        - f0_corr: Pearson correlation on aligned voiced pairs
        - vuv: Voiced/Unvoiced mismatch % on original frame grid

        Frames are 10 ms by default; we require both:
        - at least `min_voiced_ms` of aligned voiced pairs, and
        - aligned voiced pairs cover at least `min_coverage_ratio` of the overlap length.
        """
        try:
            ref, sr_ref = librosa.load(reference_path, sr=None)
            syn, sr_syn = librosa.load(synthesized_path, sr=None)
            sr = sr_ref if sr_ref == sr_syn else min(sr_ref, sr_syn)
            if sr_ref != sr:
                ref = librosa.resample(ref, orig_sr=sr_ref, target_sr=sr)
            if sr_syn != sr:
                syn = librosa.resample(syn, orig_sr=sr_syn, target_sr=sr)

            hop = int(0.010 * sr)  # 10 ms
            f0_r, vuv_r = self._extract_f0(ref, sr, hop)
            f0_s, vuv_s = self._extract_f0(syn, sr, hop)

            # V/UV mismatch on original grid (treat NaN as unvoiced)
            vuv_r_b = np.where(np.isnan(f0_r), False, vuv_r)
            vuv_s_b = np.where(np.isnan(f0_s), False, vuv_s)
            L_overlap = min(len(vuv_r_b), len(vuv_s_b))
            vuv_err = float(np.mean(vuv_r_b[:L_overlap] != vuv_s_b[:L_overlap]) * 100.0)

            # Build voiced-only sequences in log domain
            idx_r = np.where(vuv_r & (~np.isnan(f0_r)))[0]
            idx_s = np.where(vuv_s & (~np.isnan(f0_s)))[0]
            if idx_r.size < 2 or idx_s.size < 2:
                return {"gpe": float("nan"), "f0_rmse_hz": float("nan"), "f0_corr": float("nan"), "vuv": vuv_err}

            ref_voiced_hz = f0_r[idx_r]
            syn_voiced_hz = f0_s[idx_s]
            ref_log = np.log(ref_voiced_hz)
            syn_log = np.log(syn_voiced_hz)

            # DTW on 1D log-F0
            try:
                D, wp = librosa.sequence.dtw(ref_log[np.newaxis, :], syn_log[np.newaxis, :], metric="euclidean")
                wp = np.array(wp)[::-1]  # (L, 2) pairs (ref_idx, syn_idx)
                aligned_ref_hz = ref_voiced_hz[wp[:, 0]]
                aligned_syn_hz = syn_voiced_hz[wp[:, 1]]
            except Exception:
                if FASTDTW_AVAILABLE:
                    _, path = fastdtw(ref_log, syn_log, dist=lambda a, b: abs(a - b))
                    path = np.array(path, dtype=int)
                    aligned_ref_hz = ref_voiced_hz[path[:, 0]]
                    aligned_syn_hz = syn_voiced_hz[path[:, 1]]
                else:
                    L = min(ref_voiced_hz.size, syn_voiced_hz.size)
                    aligned_ref_hz = ref_voiced_hz[:L]
                    aligned_syn_hz = syn_voiced_hz[:L]

            # Drop any NaNs (shouldn't be present for voiced-only, but be safe)
            mask = (~np.isnan(aligned_ref_hz)) & (~np.isnan(aligned_syn_hz))
            r = aligned_ref_hz[mask]
            s = aligned_syn_hz[mask]
            aligned_pairs = r.size

            # ---- Minimum voiced-content guards ----
            # 1) absolute duration
            min_pairs_abs = max(2, int(round(min_voiced_ms / 10.0)))  # 10 ms per frame
            # 2) relative coverage against the overlapped frame count on original grid
            min_pairs_rel = int(np.ceil(min_coverage_ratio * max(1, L_overlap)))
            min_needed = max(min_pairs_abs, min_pairs_rel)

            if aligned_pairs < min_needed:
                # Not enough voiced alignment -> unreliable; return NaNs but keep V/UV
                return {"gpe": float("nan"), "f0_rmse_hz": float("nan"), "f0_corr": float("nan"), "vuv": vuv_err}

            # ---- Metrics on aligned voiced pairs ----
            rel_err = np.abs(r - s) / (r + 1e-8)
            gpe = float(np.mean(rel_err > 0.20) * 100.0)
            f0_rmse_hz = float(np.sqrt(np.mean((r - s) ** 2)))
            f0_corr = float(np.corrcoef(r, s)[0, 1]) if (np.std(r) > 1e-6 and np.std(s) > 1e-6) else float("nan")

            return {"gpe": gpe, "f0_rmse_hz": f0_rmse_hz, "f0_corr": f0_corr, "vuv": vuv_err}

        except Exception as e:
            logger.warning(f"Failed to compute pitch metrics: {e}")
            return {"gpe": float("nan"), "f0_rmse_hz": float("nan"), "f0_corr": float("nan"), "vuv": float("nan")}

    # ---------- Convenience wrapper ----------
    def compute_all_metrics(
        self, ref_audio: np.ndarray, syn_audio: np.ndarray, ref_text: str, sr: int = 16000
    ) -> Dict[str, float]:
        """
        Convenience path. Writes temp WAVs to reuse path-based metrics.
        """
        metrics = {}
        if sr != 16000:
            ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=16000)
            syn_audio = librosa.resample(syn_audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # ensure mono
        ref_audio = _stereo_to_mono(ref_audio)
        syn_audio = _stereo_to_mono(syn_audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_tmp, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as syn_tmp:
            sf.write(ref_tmp.name, ref_audio.astype(np.float32), sr)
            sf.write(syn_tmp.name, syn_audio.astype(np.float32), sr)
            ref_path = ref_tmp.name
            syn_path = syn_tmp.name

        try:
            mcd_val = self.compute_mcd(ref_path, syn_path)
            metrics["mcd"] = mcd_val

            # Always also compute LSD mel dB for robustness/interpretability
            metrics["lsd_mel_db"] = self.compute_lsd_mel_db(ref_path, syn_path)

            wer_pack = self.compute_wer_and_norm_with_transcript(ref_text, syn_path)
            metrics["wer"] = wer_pack["wer"]
            metrics["wer_norm"] = wer_pack["wer_norm"]
            metrics["cer"] = wer_pack["cer"]
            metrics["cer_norm"] = wer_pack["cer_norm"]
            # If you want the transcripts/normalized strings, you can return/store them too:
            # metrics["hyp"] = wer_pack["hyp"]; metrics["ref_norm"] = wer_pack["ref_norm"]; metrics["hyp_norm"] = wer_pack["hyp_norm"]

            metrics["secs"] = self.compute_secs(ref_path, syn_path)

            pitch = self.compute_pitch_metrics(ref_path, syn_path)
            metrics.update(pitch)  # adds gpe, f0_rmse_hz, f0_corr, vuv
        finally:
            try:
                os.unlink(ref_path); os.unlink(syn_path)
            except Exception:
                pass

        return metrics


# ---------- Local smoke test ----------
def test_metrics():
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0))
    ref_audio = np.sin(2 * np.pi * 220 * t) * 0.5
    syn_audio = np.sin(2 * np.pi * 230 * t) * 0.5 + np.random.normal(0, 0.02, len(t))
    ref_text = "Ceci est une phrase de test pour l'évaluation."

    computer = MetricsComputer(language="fr")
    metrics = computer.compute_all_metrics(ref_audio, syn_audio, ref_text, sr)
    print("Computed metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    # Sanity:
    # - If WORLD+pysptk present, MCD should be single-digit/low double-digit dB (not hundreds).
    # - GPE is %; f0_rmse_hz ~ few Hz for close tones; f0_corr ~ high; vuv small on pure tones.


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_metrics()
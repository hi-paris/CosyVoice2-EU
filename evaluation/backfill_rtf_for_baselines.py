#!/usr/bin/env python3
"""
Backfill RTF for baseline metrics CSVs by reading saved audio files.

For each missing/NaN RTF, compute:
  RTF = synthesis_time_seconds / generated_duration_seconds

Assumptions:
- Metrics CSVs are named: baseline_<model>_<lang>_metrics.csv (e.g., coqui, elevenlabs)
- Audio files were saved to: <synth_dir>/baseline_<model>_<lang>/<utterance_id>.wav

Usage:
    python backfill_rtf_baseline_coqui.py \
    --results_dir /path/to/results \
    --synth_dir /path/to/synthesis_output \
        --languages fr,de \
        --model coqui
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Prefer soundfile (fast) then fallback to torchaudio
try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None  # type: ignore

try:
    import torchaudio  # type: ignore
except Exception:
    torchaudio = None  # type: ignore


def audio_duration_seconds(wav_path: Path) -> float:
    if sf is not None:
        try:
            info = sf.info(str(wav_path))
            if info.frames and info.samplerate:
                return float(info.frames) / float(info.samplerate)
        except Exception:
            pass
    if torchaudio is not None:
        try:
            si = torchaudio.info(str(wav_path))
            if si.num_frames and si.sample_rate:
                return float(si.num_frames) / float(si.sample_rate)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read audio info: {wav_path}")


def backfill_for_language(results_dir: Path, synth_dir: Path, lang: str, model: str) -> Path:
    csv_path = results_dir / f"baseline_{model}_{lang}_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "rtf" not in df.columns:
        df["rtf"] = np.nan
    if "synthesis_time" not in df.columns:
        raise ValueError("CSV lacks 'synthesis_time' column; cannot compute RTF.")
    if "utterance_id" not in df.columns:
        raise ValueError("CSV lacks 'utterance_id' column; cannot map to audio files.")

    audio_root = synth_dir / f"baseline_{model}_{lang}"
    missing = df["rtf"].isna() | ~np.isfinite(df["rtf"]) | (df["rtf"] <= 0)
    updated = 0
    for idx, row in df[missing].iterrows():
        utt = str(row["utterance_id"]) if not pd.isna(row["utterance_id"]) else None
        st = float(row["synthesis_time"]) if not pd.isna(row["synthesis_time"]) else None
        if not utt or st is None or st <= 0:
            continue
        wav_path = audio_root / f"{utt}.wav"
        if not wav_path.exists():
            # skip silently; maybe audio wasn't saved
            continue
        try:
            dur = audio_duration_seconds(wav_path)
            if dur > 0:
                df.at[idx, "rtf"] = st / dur
                updated += 1
        except Exception:
            # Skip unreadable files
            continue

    out_path = csv_path  # overwrite in place
    df.to_csv(out_path, index=False)
    print(f"{lang.upper()}: updated {updated} rows with RTF -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Backfill RTF for baseline Coqui metrics CSVs")
    ap.add_argument("--results_dir", required=True, help="Folder with baseline_coqui_<lang>_metrics.csv")
    ap.add_argument("--synth_dir", required=True, help="Folder with baseline_coqui_<lang>/<utterance_id>.wav")
    ap.add_argument("--languages", default="fr,de", help="Comma-separated languages (default: fr,de)")
    ap.add_argument("--model", default="coqui", help="Baseline model name in filenames (e.g., coqui, elevenlabs)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    synth_dir = Path(args.synth_dir)
    langs = [s.strip().lower() for s in args.languages.split(",") if s.strip()]

    for lang in langs:
        try:
            backfill_for_language(results_dir, synth_dir, lang, args.model)
        except Exception as e:
            print(f"{lang.upper()}: backfill failed: {e}")


if __name__ == "__main__":
    main()

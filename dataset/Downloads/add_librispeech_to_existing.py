import os
import random
import shutil
from pathlib import Path
import librosa
import soundfile as sf
from datasets import load_dataset, Audio
import multiprocessing
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def process_mls_french(output_root, splits):
    ds = load_dataset(
        "facebook/multilingual_librispeech",
        "french",
        cache_dir="/tsi/hi-paris/tts/Luka/hf_cache"
    )
    logger.info("Loaded MLS French dataset: splits = %s", list(ds.keys()))
    
    entries = []
    
    for split in ["train", "dev", "test"]:
        # disable all on-the-fly torchcodec / ffmpeg audio decoding
        ds[split] = ds[split].cast_column(
            "audio",
            Audio(sampling_rate=16_000)
        )
        subset = ds[split]
        split_ratio = {
            "train": "train",
            "dev": "dev", 
            "test": "test"
        }[split]
        
        for item in tqdm(subset, desc=f"Processing MLS-{split}"):
            
            try:
                # Decode audio array using Hugging Face's built-in loader
                audio_info = item["audio"]
                y = audio_info["array"]
                original_sr = audio_info["sampling_rate"]
                sr = 16000
                # Resample if necessary (cast_column already set sampling_rate)
                if original_sr != sr:
                    y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)
                
                # Process the rest as before
                utt_id = item["id"]  # like "10148_10119_000000"
                speaker = item["speaker_id"]
                chapter = item["chapter_id"]
                
                # IMPORTANT: Add libri_ prefix to speaker to distinguish from EmoNet
                target_dir = Path(output_root) / split / f"libri_{speaker}" / str(chapter)
                target_dir.mkdir(parents=True, exist_ok=True)
                
                final_id = f"libri_{speaker}_{split}_{utt_id}"
                wav_out = target_dir / f"{final_id}.wav"
                txt_out = target_dir / f"{final_id}.normalized.txt"
                
                # Save audio
                sf.write(str(wav_out), y, sr)
                
                # Normalize text (simple strip) - use transcript field
                text = item["transcript"].strip().replace("\n", " ")
                
                with open(txt_out, "w", encoding="utf-8") as f:
                    f.write(text)
                
                entries.append((split, wav_out, txt_out))
                
            except Exception as e:
                logger.error(f"Error processing item {item.get('id', 'unknown')}: {e}")
                continue
    
    logger.info("Processed %d MLS French utterances", len(entries))
    return entries

def main():
    # YOUR PATHS
    root = "/tsi/hi-paris/tts/Luka/tts_dataset_combined"  # Your existing combined dataset
    splits = {"train": .9, "dev": .05, "test": .05}
    
    # Ensure output directory exists
    Path(root).mkdir(parents=True, exist_ok=True)
    
    multiprocessing.set_start_method('spawn', force=True)
    
    logger.info("Adding MLS French data to existing dataset...")
    logger.info(f"Target directory: {root}")
    
    # Only process LibriSpeech - skip EmoNet since it's already there
    process_mls_french(root, splits)
    
    logger.info("✅ Added LibriSpeech data to existing combined dataset")
    logger.info(f"✅ Combined dataset location: {root}")

if __name__ == "__main__":
    main()

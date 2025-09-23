#!/usr/bin/env python3
"""
Direct Dataset Reader for CosyVoice2 Evaluation

Reads directly from your file structure without requiring data conversion.
Structure: split -> speaker_id -> audio_id -> {audio.wav, transcript.txt} pairs
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DatasetReader:
    """Read dataset directly from file structure."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {base_path}")
    
    def get_samples(self, splits: List[str], max_samples_per_split: Optional[int] = None) -> List[Dict]:
        """
        Get all audio/text pairs from specified splits.
        
        Returns:
            List of dicts with keys: utterance_id, split, speaker_id, audio_id, 
                                   wav_path, txt_path, text
        """
        samples = []
        
        for split in splits:
            split_path = self.base_path / split
            if not split_path.exists():
                logger.warning(f"Split directory not found: {split_path}")
                continue
                
            logger.info(f"Processing {split} split...")
            split_samples = []
            
            # Iterate through speakers
            for speaker_dir in split_path.iterdir():
                if not speaker_dir.is_dir():
                    continue
                    
                speaker_id = speaker_dir.name
                
                # Iterate through audio_ids
                for audio_dir in speaker_dir.iterdir():
                    if not audio_dir.is_dir():
                        continue
                        
                    audio_id = audio_dir.name
                    
                    # Find wav/txt pairs in this directory
                    wav_files = list(audio_dir.glob("*.wav"))
                    
                    for wav_file in wav_files:
                        # Find corresponding text file
                        txt_file = wav_file.with_suffix(".normalized.txt")
                        if not txt_file.exists():
                            logger.warning(f"Text file not found for {wav_file}")
                            continue
                        
                        # Read text content
                        try:
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                        except Exception as e:
                            logger.error(f"Error reading {txt_file}: {e}")
                            continue
                        
                        # Create utterance ID
                        utterance_id = f"{split}_{speaker_id}_{audio_id}_{wav_file.stem}"
                        
                        sample = {
                            'utterance_id': utterance_id,
                            'split': split,
                            'speaker_id': speaker_id,
                            'audio_id': audio_id,
                            'wav_path': str(wav_file),
                            'txt_path': str(txt_file),
                            'text': text
                        }
                        split_samples.append(sample)
            
            # Apply limit if specified (robust to string/None/zero)
            limit = max_samples_per_split
            # Coerce string digits to int
            if isinstance(limit, str):
                try:
                    limit = int(limit)
                except ValueError:
                    limit = None
            # Only apply if positive int
            if isinstance(limit, int) and limit > 0 and len(split_samples) > limit:
                split_samples = split_samples[:limit]
                logger.info(f"Limited {split} to {limit} samples")
            
            samples.extend(split_samples)
            logger.info(f"Found {len(split_samples)} samples in {split}")
        
        logger.info(f"Total samples: {len(samples)}")
        return samples
    
    def get_speaker_stats(self, splits: List[str]) -> Dict:
        """Get speaker statistics from the dataset."""
        speaker_counts = {}
        
        for split in splits:
            split_path = self.base_path / split
            if not split_path.exists():
                continue
                
            for speaker_dir in split_path.iterdir():
                if not speaker_dir.is_dir():
                    continue
                    
                speaker_id = speaker_dir.name
                
                # Count audio files for this speaker
                audio_count = 0
                for audio_dir in speaker_dir.iterdir():
                    if audio_dir.is_dir():
                        audio_count += len(list(audio_dir.glob("*.wav")))
                
                if speaker_id not in speaker_counts:
                    speaker_counts[speaker_id] = {}
                speaker_counts[speaker_id][split] = audio_count
        
        return speaker_counts


def test_dataset_reader():
    """Test the dataset reader."""
    base_path = "/tsi/hi-paris/tts/Luka/data/tts_dataset_combined"
    
    reader = DatasetReader(base_path)
    
    # Test with limited samples
    samples = reader.get_samples(["dev"], max_samples_per_split=5)
    
    print(f"Found {len(samples)} samples")
    for sample in samples[:3]:
        print(f"  {sample['utterance_id']}: {sample['text'][:50]}...")
    
    # Get speaker stats
    stats = reader.get_speaker_stats(["dev", "test"])
    print(f"\nSpeaker statistics:")
    for speaker, counts in list(stats.items())[:5]:
        print(f"  {speaker}: {counts}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataset_reader()

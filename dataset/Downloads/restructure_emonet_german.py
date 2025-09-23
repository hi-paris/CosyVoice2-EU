import os
import shutil
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def restructure_emonet_german():
    """Restructure German EmoNet data to match the existing structure with DE suffixes."""
    base_dir = "/tsi/hi-paris/tts/Luka/data/EmoNet_german_split"
    splits = ["train", "dev", "test"]
    
    logger.info(f"Restructuring German EmoNet data from: {base_dir}")
    
    for split in splits:
        split_dir = Path(base_dir) / split
        if not split_dir.exists():
            logger.warning(f"Split directory does not exist: {split_dir}")
            continue
            
        logger.info(f"Processing {split} split...")
        files = [f for f in split_dir.iterdir() if f.suffix in [".wav", ".txt", ".normalized.txt"]]
        
        logger.info(f"Found {len(files)} files in {split}")
        
        for f in tqdm(files, desc=f"Restructuring {split}"):
            name = f.stem
            parts = name.split("_")
            if len(parts) < 3:
                logger.warning(f"Skipping file with insufficient parts: {name}")
                continue
            
            # Extract speaker name (first part)
            speaker = parts[0]
            
            # Find hash segment (the one with 8 hex chars) or use a unique identifier
            hash_id = None
            for p in parts:
                if len(p) == 8 and p.isalnum() and all(c in '0123456789abcdefABCDEF' for c in p):
                    hash_id = p
                    break
            
            if not hash_id:
                # If no hash found, use the last part as hash_id
                hash_id = parts[-1] if parts[-1] else "unknown"
            
            # Create new directory structure with DE suffix
            new_dir = split_dir / f"{speaker}_DE" / hash_id
            new_dir.mkdir(parents=True, exist_ok=True)
            
            # Move file
            try:
                shutil.move(str(f), new_dir / f.name)
            except Exception as e:
                logger.error(f"Failed to move {f}: {e}")
                continue
    
    logger.info("✅ German EmoNet restructuring complete!")

if __name__ == "__main__":
    restructure_emonet_german()

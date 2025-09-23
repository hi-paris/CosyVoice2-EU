import os
import shutil
from pathlib import Path
from tqdm import tqdm

base_dir = "/tsi/hi-paris/tts/Luka/data/EmoNet_split"
splits = ["train", "dev", "test"]

for split in splits:
    split_dir = Path(base_dir) / split
    files = [f for f in split_dir.iterdir() if f.suffix in [".wav", ".txt", ".normalized.txt"]]
    
    for f in tqdm(files, desc=f"Restructuring {split}"):
        name = f.stem
        parts = name.split("_")
        if len(parts) < 5:
            continue
        
        speaker = parts[0]
        # find hash segment (the one with 8 hex chars)
        hash_id = next((p for p in parts if len(p) == 8 and p.isalnum()), "unknown")
        
        new_dir = split_dir / speaker / hash_id
        new_dir.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(str(f), new_dir / f.name)
#!/usr/bin/env python3
"""
Master script to add German data to existing TTS dataset.

This script:
1. Creates a backup of the existing dataset
2. Downloads German EmoNet data
3. Downloads German LibriSpeech data
4. Restructures the data to match existing format
5. Merges everything into the combined dataset with DE suffixes

Usage:
    python add_german_data_to_existing.py [--skip-backup] [--skip-emonet] [--skip-librispeech]
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
import shutil

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description, check=True):
    """Run a command and log the output."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False

def create_backup():
    """Create a backup of the existing dataset."""
    logger.info("Creating backup of existing dataset...")
    
    backup_script = "backup_tts_dataset.sh"
    if not os.path.exists(backup_script):
        logger.error(f"Backup script not found: {backup_script}")
        return False
    
    # Make sure the script is executable
    os.chmod(backup_script, 0o755)
    
    success = run_command(["./backup_tts_dataset.sh"], "Creating backup")
    if success:
        logger.info("✅ Backup completed successfully")
    else:
        logger.error("❌ Backup failed")
    
    return success

def download_emonet_german():
    """Download and process German EmoNet data."""
    logger.info("Downloading German EmoNet data...")
    
    script_path = "dataset/Downloads/download_emonet_german.py"
    if not os.path.exists(script_path):
        logger.error(f"German EmoNet script not found: {script_path}")
        return False
    
    success = run_command([sys.executable, script_path], "Downloading German EmoNet data")
    if success:
        logger.info("✅ German EmoNet download completed")
    else:
        logger.error("❌ German EmoNet download failed")
    
    return success

def restructure_emonet_german():
    """Restructure German EmoNet data to match existing format."""
    logger.info("Restructuring German EmoNet data...")
    
    script_path = "dataset/Downloads/restructure_emonet_german.py"
    if not os.path.exists(script_path):
        logger.error(f"German EmoNet restructuring script not found: {script_path}")
        return False
    
    success = run_command([sys.executable, script_path], "Restructuring German EmoNet data")
    if success:
        logger.info("✅ German EmoNet restructuring completed")
    else:
        logger.error("❌ German EmoNet restructuring failed")
    
    return success

def download_librispeech_german():
    """Download and process German LibriSpeech data."""
    logger.info("Downloading German LibriSpeech data...")
    
    script_path = "dataset/Downloads/add_librispeech_german_to_existing.py"
    if not os.path.exists(script_path):
        logger.error(f"German LibriSpeech script not found: {script_path}")
        return False
    
    success = run_command([sys.executable, script_path], "Downloading German LibriSpeech data")
    if success:
        logger.info("✅ German LibriSpeech download completed")
    else:
        logger.error("❌ German LibriSpeech download failed")
    
    return success

def merge_german_emonet_to_combined():
    """Merge German EmoNet data into the combined dataset."""
    logger.info("Merging German EmoNet data into combined dataset...")
    
    source_dir = "/tsi/hi-paris/tts/Luka/data/EmoNet_german_split"
    target_dir = "/tsi/hi-paris/tts/Luka/data/tts_dataset_combined"
    
    if not os.path.exists(source_dir):
        logger.error(f"Source directory does not exist: {source_dir}")
        return False
    
    if not os.path.exists(target_dir):
        logger.error(f"Target directory does not exist: {target_dir}")
        return False
    
    # Copy German EmoNet data to combined dataset
    for split in ["train", "dev", "test"]:
        source_split = os.path.join(source_dir, split)
        target_split = os.path.join(target_dir, split)
        
        if not os.path.exists(source_split):
            logger.warning(f"Source split does not exist: {source_split}")
            continue
        
        logger.info(f"Copying {split} split...")
        
        # Copy all speaker directories with DE suffix
        for speaker_dir in os.listdir(source_split):
            if not speaker_dir.endswith("_DE"):
                continue
                
            source_speaker = os.path.join(source_split, speaker_dir)
            target_speaker = os.path.join(target_split, speaker_dir)
            
            if os.path.isdir(source_speaker):
                if os.path.exists(target_speaker):
                    logger.warning(f"Target speaker directory already exists, skipping: {target_speaker}")
                    continue
                
                try:
                    shutil.copytree(source_speaker, target_speaker)
                    logger.info(f"Copied {speaker_dir} to {split}")
                except Exception as e:
                    logger.error(f"Failed to copy {speaker_dir}: {e}")
                    return False
    
    logger.info("✅ German EmoNet data merged successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description="Add German data to existing TTS dataset")
    parser.add_argument("--skip-backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--skip-emonet", action="store_true", help="Skip German EmoNet processing")
    parser.add_argument("--skip-librispeech", action="store_true", help="Skip German LibriSpeech processing")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting German data addition process...")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual changes will be made")
        logger.info("Would execute the following steps:")
        if not args.skip_backup:
            logger.info("  1. Create backup of existing dataset")
        if not args.skip_emonet:
            logger.info("  2. Download German EmoNet data")
            logger.info("  3. Restructure German EmoNet data")
            logger.info("  4. Merge German EmoNet data to combined dataset")
        if not args.skip_librispeech:
            logger.info("  5. Download German LibriSpeech data")
        return
    
    # Step 1: Create backup (unless skipped)
    if not args.skip_backup:
        if not create_backup():
            logger.error("Backup failed. Aborting.")
            return
    else:
        logger.info("Skipping backup as requested")
    
    # Step 2: Download and process German EmoNet (unless skipped)
    if not args.skip_emonet:
        # if not download_emonet_german():
        #     logger.error("German EmoNet download failed. Aborting.")
        #     return
        
        # if not restructure_emonet_german():
        #     logger.error("German EmoNet restructuring failed. Aborting.")
        #     return
        
        if not merge_german_emonet_to_combined():
            logger.error("German EmoNet merge failed. Aborting.")
            return
    else:
        logger.info("Skipping German EmoNet processing as requested")
    
    # Step 3: Download and process German LibriSpeech (unless skipped)
    if not args.skip_librispeech:
        if not download_librispeech_german():
            logger.error("German LibriSpeech download failed. Aborting.")
            return
    else:
        logger.info("Skipping German LibriSpeech processing as requested")
    
    logger.info("🎉 German data addition process completed successfully!")
    logger.info("Your combined dataset now includes German data with DE suffixes.")
    logger.info("Dataset location: /tsi/hi-paris/tts/Luka/data/tts_dataset_combined")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Robust backup and filter for EmoNet dataset using rsync.
Improved version with better error handling and efficiency.
"""

import subprocess
import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import logging

# Import soundfile at module level for efficiency
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available, audio duration checks will be skipped")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('backup_filter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def verify_backup(src_dir, dst_dir):
    """Verify backup integrity by comparing file counts."""
    try:
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        
        if not dst_path.exists():
            return False
            
        # Count files in both directories
        src_files = len(list(src_path.rglob('*')))
        dst_files = len(list(dst_path.rglob('*')))
        
        logger.info(f"Source files: {src_files}, Backup files: {dst_files}")
        
        # Allow for small differences due to timing
        if abs(src_files - dst_files) <= 5:
            logger.info("Backup verification: PASSED")
            return True
        else:
            logger.error("Backup verification: FAILED - file count mismatch")
            return False
            
    except Exception as e:
        logger.error(f"Backup verification failed: {e}")
        return False

def fast_backup(src_dir, dst_dir):
    """Create backup using rsync - much faster for large datasets."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    if not src_path.exists():
        logger.error(f"Source directory does not exist: {src_dir}")
        return False
        
    logger.info(f"Creating backup: {src_dir} -> {dst_dir}")
    logger.info("Using rsync for fast backup...")
    
    try:
        # Ensure destination parent directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rsync with progress and checksum verification
        cmd = [
            'rsync', '-avc', '--progress', '--stats',
            f'{src_dir}/', f'{dst_dir}/'
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        logger.info("Rsync output:")
        logger.info(result.stdout)
        
        # Verify backup
        if verify_backup(src_dir, dst_dir):
            logger.info("Backup completed and verified successfully!")
            return True
        else:
            logger.error("Backup verification failed!")
            return False
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Backup failed: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.warning("rsync not found. Falling back to basic copy...")
        return basic_backup(src_dir, dst_dir)

def basic_backup(src_dir, dst_dir):
    """Fallback backup using cp command."""
    try:
        dst_path = Path(dst_dir)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = ['cp', '-r', str(src_dir), str(dst_dir)]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        if verify_backup(src_dir, dst_dir):
            logger.info("Basic backup completed and verified!")
            return True
        else:
            logger.error("Basic backup verification failed!")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Basic backup failed: {e}")
        return False

def get_audio_duration(wav_path):
    """Get audio duration efficiently."""
    if not SOUNDFILE_AVAILABLE:
        return None
        
    try:
        info = sf.info(wav_path)
        return info.frames / info.samplerate
    except Exception:
        return None

def apply_filters(data_dir, dry_run=False, stats_file=None):
    """Apply filtering with robust error handling and statistics."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return False
    
    filters = {
        'min_duration': 1.0,
        'max_duration': 30.0,
        'min_text_chars': 3,
    }
    
    logger.info(f"Applying filters: {filters}")
    if dry_run:
        logger.info("DRY RUN MODE - no files will be deleted")
    
    total_processed = 0
    total_removed = 0
    stats = {}
    
    for split in ['train', 'dev', 'test']:
        split_dir = data_path / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
            
        logger.info(f"Processing {split.upper()} split...")
        
        # Get all wav files
        wav_files = list(split_dir.rglob('*.wav'))
        logger.info(f"Found {len(wav_files)} wav files in {split}")
        
        if not wav_files:
            logger.warning(f"No wav files found in {split}")
            continue
        
        # Process files
        to_remove = []
        split_stats = {
            'total_files': len(wav_files),
            'no_text': 0,
            'short_text': 0,
            'short_audio': 0,
            'long_audio': 0,
            'audio_error': 0,
            'removed': 0
        }
        
        logger.info("Analyzing files...")
        for wav_path in tqdm(wav_files, desc=f"Analyzing {split}"):
            total_processed += 1
            should_remove = False
            removal_reason = None
            
            base_name = wav_path.stem
            
            # Check for text file
            txt_path = None
            for txt_name in [f"{base_name}.normalized.txt", f"{base_name}.txt"]:
                potential_txt = wav_path.parent / txt_name
                if potential_txt.exists():
                    txt_path = potential_txt
                    break
            
            # Text validation
            if not txt_path:
                should_remove = True
                removal_reason = "no_text"
                split_stats['no_text'] += 1
            else:
                try:
                    # Efficient text check
                    file_size = txt_path.stat().st_size
                    if file_size < filters['min_text_chars']:
                        should_remove = True
                        removal_reason = "short_text"
                        split_stats['short_text'] += 1
                    else:
                        # Read and validate content
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            content = f.read(200).strip()  # Read up to 200 chars
                        if len(content) < filters['min_text_chars']:
                            should_remove = True
                            removal_reason = "short_text"
                            split_stats['short_text'] += 1
                except Exception as e:
                    logger.debug(f"Error reading text file {txt_path}: {e}")
                    should_remove = True
                    removal_reason = "text_error"
                    split_stats['short_text'] += 1
            
            # Audio validation (only if text is OK)
            if not should_remove and SOUNDFILE_AVAILABLE:
                duration = get_audio_duration(wav_path)
                if duration is None:
                    should_remove = True
                    removal_reason = "audio_error"
                    split_stats['audio_error'] += 1
                elif duration < filters['min_duration']:
                    should_remove = True
                    removal_reason = "short_audio"
                    split_stats['short_audio'] += 1
                elif duration > filters['max_duration']:
                    should_remove = True
                    removal_reason = "long_audio"
                    split_stats['long_audio'] += 1
            
            if should_remove:
                to_remove.append((wav_path, txt_path, removal_reason))
        
        # Remove files
        split_stats['removed'] = len(to_remove)
        total_removed += len(to_remove)
        
        if to_remove and not dry_run:
            logger.info(f"Removing {len(to_remove)} files from {split}...")
            for wav_path, txt_path, reason in tqdm(to_remove, desc="Removing files"):
                try:
                    wav_path.unlink(missing_ok=True)
                    if txt_path and txt_path.exists():
                        txt_path.unlink(missing_ok=True)
                    
                    # Clean up any other text variants
                    base_name = wav_path.stem
                    for txt_variant in [
                        wav_path.parent / f"{base_name}.normalized.txt",
                        wav_path.parent / f"{base_name}.txt"
                    ]:
                        if txt_variant.exists() and txt_variant != txt_path:
                            txt_variant.unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Error removing files for {wav_path}: {e}")
        
        # Log statistics
        removed_pct = (split_stats['removed'] / split_stats['total_files']) * 100
        logger.info(f"{split}: {split_stats['removed']}/{split_stats['total_files']} files removed ({removed_pct:.1f}%)")
        
        stats[split] = split_stats
    
    # Save detailed statistics
    if stats_file:
        with open(stats_file, 'w') as f:
            json.dump({
                'filters': filters,
                'total_processed': total_processed,
                'total_removed': total_removed,
                'split_stats': stats,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        logger.info(f"Statistics saved to: {stats_file}")
    
    logger.info(f"Filtering complete: {total_removed}/{total_processed} files removed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Robust backup and filter for EmoNet dataset")
    parser.add_argument('data_dir', help="Path to EmoNet_split directory")
    parser.add_argument('--backup-dir', help="Backup directory (default: auto-generated)")
    parser.add_argument('--skip-backup', action='store_true', help="Skip backup step")
    parser.add_argument('--backup-only', action='store_true', help="Only backup, no filtering")
    parser.add_argument('--dry-run', action='store_true', help="Dry run mode (no actual changes)")
    parser.add_argument('--filter-only', action='store_true', help="Only filter, no backup")
    parser.add_argument('--stats-file', help="Save filtering statistics to JSON file")
    parser.add_argument('--force-continue', action='store_true', help="Continue even if backup fails")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    # Set up backup directory
    if args.backup_dir:
        backup_dir = Path(args.backup_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = data_dir.parent / f"{data_dir.name}_backup_{timestamp}"
    
    # Backup step
    backup_success = True
    if not args.skip_backup and not args.filter_only:
        logger.info("=" * 60)
        logger.info("BACKUP PHASE")
        logger.info("=" * 60)
        backup_success = fast_backup(data_dir, backup_dir)
        
        if not backup_success:
            logger.error("Backup failed!")
            if not args.force_continue:
                logger.error("Stopping execution. Use --force-continue to proceed anyway.")
                sys.exit(1)
            else:
                logger.warning("Continuing despite backup failure due to --force-continue")
    
    # Filter step
    if not args.backup_only:
        logger.info("\n" + "=" * 60)
        logger.info("FILTERING PHASE")
        logger.info("=" * 60)
        
        stats_file = args.stats_file or f"filter_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filter_success = apply_filters(data_dir, dry_run=args.dry_run, stats_file=stats_file)
        
        if not filter_success:
            logger.error("Filtering failed!")
            sys.exit(1)
    else:
        logger.info("Skipping filtering due to --backup-only flag")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("Dry run completed successfully. Remove --dry-run to apply changes.")
    else:
        if backup_success:
            logger.info(f"✅ Backup: {backup_dir}")
        if not args.backup_only:
            logger.info(f"✅ Filtering: Complete")
            logger.info(f"✅ Statistics: {stats_file}")
        else:
            logger.info("ℹ️ Filtering: Skipped (backup-only mode)")
    
    logger.info("All operations completed successfully!")

if __name__ == '__main__':
    main()

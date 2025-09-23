#!/usr/bin/env python3
"""
Efficient filtering script using the pre-computed audio_duration_mapping.csv
Filters German EmoNet files based on already computed metadata.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import logging

def setup_logging(log_file=None):
    if log_file is None:
        log_file = f"csv_filter_german_emonet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_and_analyze_csv(csv_file):
    """Load the CSV and identify files to filter."""
    logger.info(f"Loading audio duration mapping from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} file entries from CSV")
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        return None, None
    
    # Filter for German EmoNet files only
    german_emonet = df[
        (df['language'] == 'DE') & 
        (df['dataset_type'] == 'EmoNet')
    ].copy()
    
    logger.info(f"Found {len(german_emonet)} German EmoNet files in the dataset")
    
    if len(german_emonet) == 0:
        logger.warning("No German EmoNet files found in the CSV!")
        return df, None
    
    # Define filter criteria
    filters = {
        'min_duration': 1.0,
        'max_duration': 30.0,
        'min_text_chars': 3,
    }
    
    logger.info(f"Applying filter criteria: {filters}")
    
    # Create filter conditions
    conditions = []
    filter_reasons = []
    
    # Duration filters
    short_audio = german_emonet['duration'] < filters['min_duration']
    long_audio = german_emonet['duration'] > filters['max_duration']
    
    # Text filters (char_count of 0 indicates missing/empty text)
    missing_empty_text = german_emonet['char_count'] < filters['min_text_chars']
    
    # Invalid files (is_valid=False indicates audio or text errors)
    invalid_files = german_emonet['is_valid'] == False
    
    # Combine all conditions
    files_to_remove = german_emonet[
        short_audio | long_audio | missing_empty_text | invalid_files
    ].copy()
    
    # Add removal reasons
    files_to_remove['removal_reason'] = 'unknown'
    files_to_remove.loc[short_audio[short_audio].index, 'removal_reason'] = 'short_audio'
    files_to_remove.loc[long_audio[long_audio].index, 'removal_reason'] = 'long_audio'  
    files_to_remove.loc[missing_empty_text[missing_empty_text].index, 'removal_reason'] = 'short_text'
    files_to_remove.loc[invalid_files[invalid_files].index, 'removal_reason'] = 'invalid_file'
    
    # Handle overlapping reasons (prioritize by severity)
    for idx in files_to_remove.index:
        reasons = []
        if idx in invalid_files[invalid_files].index:
            reasons.append('invalid_file')
        if idx in short_audio[short_audio].index:
            reasons.append('short_audio')
        if idx in long_audio[long_audio].index:
            reasons.append('long_audio')
        if idx in missing_empty_text[missing_empty_text].index:
            reasons.append('short_text')
        files_to_remove.loc[idx, 'removal_reason'] = '_'.join(reasons)
    
    logger.info(f"Analysis complete:")
    logger.info(f"  Total German EmoNet files: {len(german_emonet)}")
    logger.info(f"  Files to remove: {len(files_to_remove)}")
    logger.info(f"  Removal percentage: {len(files_to_remove)/len(german_emonet)*100:.1f}%")
    
    # Breakdown by reason
    logger.info("Removal reasons breakdown:")
    reason_counts = files_to_remove['removal_reason'].value_counts()
    for reason, count in reason_counts.items():
        pct = count / len(german_emonet) * 100
        logger.info(f"  {reason}: {count} files ({pct:.1f}%)")
    
    # Breakdown by split
    logger.info("Removal by split:")
    for split in ['train', 'dev', 'test']:
        split_total = len(german_emonet[german_emonet['split'] == split])
        split_remove = len(files_to_remove[files_to_remove['split'] == split])
        if split_total > 0:
            split_pct = split_remove / split_total * 100
            logger.info(f"  {split}: {split_remove}/{split_total} files ({split_pct:.1f}%)")
    
    return df, files_to_remove

def apply_filtering(files_to_remove, dry_run=False, stats_file=None):
    """Apply the filtering by removing the identified files."""
    
    if files_to_remove is None or len(files_to_remove) == 0:
        logger.info("No files to remove!")
        return True
    
    logger.info(f"{'DRY RUN: ' if dry_run else ''}Removing {len(files_to_remove)} files...")
    
    removal_stats = {
        'total_attempted': len(files_to_remove),
        'successful_removals': 0,
        'failed_removals': 0,
        'missing_files': 0,
        'by_split': {},
        'by_reason': {},
        'errors': []
    }
    
    for idx, row in tqdm(files_to_remove.iterrows(), total=len(files_to_remove), desc="Processing files"):
        file_path = Path(row['file_path'])
        split = row['split']
        reason = row['removal_reason']
        
        # Initialize stats tracking
        if split not in removal_stats['by_split']:
            removal_stats['by_split'][split] = {'attempted': 0, 'successful': 0, 'failed': 0}
        if reason not in removal_stats['by_reason']:
            removal_stats['by_reason'][reason] = {'attempted': 0, 'successful': 0, 'failed': 0}
        
        removal_stats['by_split'][split]['attempted'] += 1
        removal_stats['by_reason'][reason]['attempted'] += 1
        
        try:
            if not file_path.exists():
                logger.debug(f"File does not exist (already removed?): {file_path}")
                removal_stats['missing_files'] += 1
                continue
            
            if not dry_run:
                # Remove wav file
                file_path.unlink()
                
                # Remove associated text files
                base_name = file_path.stem
                for txt_ext in ['.normalized.txt', '.txt']:
                    txt_file = file_path.parent / f"{base_name}{txt_ext}"
                    if txt_file.exists():
                        txt_file.unlink()
            
            removal_stats['successful_removals'] += 1
            removal_stats['by_split'][split]['successful'] += 1
            removal_stats['by_reason'][reason]['successful'] += 1
            
        except Exception as e:
            error_msg = f"Failed to remove {file_path}: {e}"
            logger.error(error_msg)
            removal_stats['errors'].append(error_msg)
            removal_stats['failed_removals'] += 1
            removal_stats['by_split'][split]['failed'] += 1
            removal_stats['by_reason'][reason]['failed'] += 1
    
    # Log final statistics
    logger.info("Removal completed!")
    logger.info(f"  Successful: {removal_stats['successful_removals']}")
    logger.info(f"  Failed: {removal_stats['failed_removals']}")
    logger.info(f"  Missing files: {removal_stats['missing_files']}")
    
    if removal_stats['errors']:
        logger.warning(f"Encountered {len(removal_stats['errors'])} errors during removal")
    
    # Save detailed statistics
    if stats_file:
        detailed_stats = {
            'removal_stats': removal_stats,
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run
        }
        
        with open(stats_file, 'w') as f:
            json.dump(detailed_stats, f, indent=2)
        logger.info(f"Detailed statistics saved to: {stats_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Filter German EmoNet files using pre-computed CSV data"
    )
    parser.add_argument(
        '--csv-file',
        default='/tsi/hi-paris/tts/Luka/data/audio_duration_mapping.csv',
        help="Path to audio duration mapping CSV file"
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help="Dry run mode (analyze but don't remove files)"
    )
    parser.add_argument(
        '--stats-file', 
        help="Save detailed statistics to JSON file"
    )
    parser.add_argument(
        '--log-file',
        help="Custom log file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_file)
    
    # Validate CSV file
    csv_file = Path(args.csv_file)
    if not csv_file.exists():
        logger.error(f"CSV file does not exist: {csv_file}")
        sys.exit(1)
    
    # Set up stats file
    if not args.stats_file:
        args.stats_file = f"csv_german_emonet_filter_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Run analysis and filtering
    logger.info("="*60)
    logger.info("CSV-BASED GERMAN EMONET FILTERING")
    logger.info("="*60)
    logger.info(f"CSV file: {csv_file}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE (will delete files)'}")
    logger.info("="*60)
    
    # Load and analyze data
    df, files_to_remove = load_and_analyze_csv(csv_file)
    
    if df is None:
        logger.error("Failed to load CSV data!")
        sys.exit(1)
    
    # Apply filtering
    success = apply_filtering(files_to_remove, dry_run=args.dry_run, stats_file=args.stats_file)
    
    if not success:
        logger.error("Filtering failed!")
        sys.exit(1)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    if args.dry_run:
        logger.info("✅ Dry run completed successfully.")
        logger.info("💡 Remove --dry-run to apply the filtering.")
        if files_to_remove is not None and len(files_to_remove) > 0:
            logger.info(f"📋 {len(files_to_remove)} German EmoNet files would be removed")
    else:
        logger.info("✅ German EmoNet filtering completed successfully!")
        logger.info("💡 Run audio_duration_summary.py again to see the updated statistics.")
    
    logger.info(f"📊 Statistics saved to: {args.stats_file}")
    
    if not args.dry_run and files_to_remove is not None and len(files_to_remove) > 0:
        logger.info("\n🔄 Recommended next steps:")
        logger.info("1. Run audio_duration_summary.py to verify results")
        logger.info("2. Create new balanced sample lists with updated data")

if __name__ == '__main__':
    main()

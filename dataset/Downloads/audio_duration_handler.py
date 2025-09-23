#!/usr/bin/env python3
import os
import argparse
import json
import csv
import random
import multiprocessing
from soundfile import info
from datetime import timedelta
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import datetime

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def process_file_pair(file_pair):
    """Process both audio and text file in a single function call."""
    wav_path, txt_path = file_pair
    
    # Process audio
    audio_duration = 0.0
    audio_error = None
    try:
        meta = info(wav_path)
        audio_duration = meta.frames / meta.samplerate
    except Exception as e:
        audio_error = str(e)
    
    # Process text efficiently
    text_info = {
        'char_count': 0,
        'word_count': 0,
        'is_empty': True,
        'is_too_short': True,
        'exists': False,
        'error': None
    }
    
    if txt_path is None:
        text_info['error'] = "Text file path is None (likely due to long filename)"
        text_info['exists'] = False
    else:
        try:
            # Quick file size check first
            if os.path.getsize(txt_path) == 0:
                text_info.update({'exists': True, 'is_empty': True, 'is_too_short': True})
            else:
                # Read only first 1000 chars for efficiency in large files
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000).strip()
                
                char_count = len(content)
                word_count = len(content.split()) if content else 0
                
                text_info.update({
                    'char_count': char_count,
                    'word_count': word_count,
                    'is_empty': char_count == 0,
                    'is_too_short': char_count < 3,
                    'exists': True
                })
        except Exception as e:
            text_info['error'] = str(e)
    
    return {
        'wav_path': wav_path,
        'txt_path': txt_path,
        'audio_duration': audio_duration,
        'audio_error': audio_error,
        'text_info': text_info
    }

def categorize_file_path(file_path):
    """Extract metadata from file path."""
    path_parts = Path(file_path).parts
    
    # Find the folder that contains language and dataset info
    for part in path_parts:
        if '_FR' in part or '_DE' in part:
            # Extract language
            language = 'FR' if '_FR' in part else 'DE'
            
            # Extract dataset type and voice/speaker info
            if part.startswith('emo_'):
                dataset_type = 'EmoNet'
                voice = part.replace('_FR', '').replace('_DE', '').replace('emo_', '')
            elif part.startswith('libri_'):
                dataset_type = 'LibriSpeech'
                speaker_id = part.replace('_FR', '').replace('_DE', '').replace('libri_', '')
                voice = f"speaker_{speaker_id}"
            else:
                dataset_type = 'Unknown'
                voice = 'Unknown'
            
            return {
                'language': language,
                'dataset_type': dataset_type, 
                'voice': voice,
                'folder_name': part
            }
    
    return {'language': 'Unknown', 'dataset_type': 'Unknown', 'voice': 'Unknown', 'folder_name': 'Unknown'}

def process_file_pair_enhanced(file_pair_with_split):
    """Enhanced processing with categorization and metadata."""
    file_pair, split = file_pair_with_split
    wav_path, txt_path = file_pair
    
    # Get basic file info
    result = process_file_pair(file_pair)
    
    # Add categorization
    category_info = categorize_file_path(wav_path)
    
    # Add split info and file metadata
    result.update({
        'split': split,
        'category': category_info,
        'file_size_wav': os.path.getsize(wav_path) if os.path.exists(wav_path) else 0,
        'file_size_txt': os.path.getsize(txt_path) if txt_path and os.path.exists(txt_path) else 0,
        'last_modified': os.path.getmtime(wav_path) if os.path.exists(wav_path) else 0
    })
    
    return result

def create_file_hash(file_path, size):
    """Create a simple hash for file identification."""
    return hashlib.md5(f"{file_path}_{size}".encode()).hexdigest()[:12]

def load_existing_mapping(mapping_file):
    """Load existing mapping file if it exists."""
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing mapping file: {e}")
    return {}

def save_mapping_file(mapping_data, mapping_file):
    """Save the comprehensive mapping file."""
    try:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        print(f"Saved mapping file: {mapping_file}")
        
        # Also save a CSV version for easy inspection
        csv_file = mapping_file.replace('.json', '.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            if mapping_data:
                fieldnames = ['file_path', 'duration', 'split', 'language', 'dataset_type', 'voice', 
                             'char_count', 'word_count', 'file_size_wav', 'is_valid']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for file_path, data in mapping_data.items():
                    row = {
                        'file_path': file_path,
                        'duration': data.get('audio_duration', 0),
                        'split': data.get('split', 'Unknown'),
                        'language': data.get('category', {}).get('language', 'Unknown'),
                        'dataset_type': data.get('category', {}).get('dataset_type', 'Unknown'),
                        'voice': data.get('category', {}).get('voice', 'Unknown'),
                        'char_count': data.get('text_info', {}).get('char_count', 0),
                        'word_count': data.get('text_info', {}).get('word_count', 0),
                        'file_size_wav': data.get('file_size_wav', 0),
                        'is_valid': not (data.get('audio_error') or data.get('text_info', {}).get('error'))
                    }
                    writer.writerow(row)
        print(f"Saved CSV mapping file: {csv_file}")
        
    except Exception as e:
        print(f"Error saving mapping file: {e}")

def aggregate_by_categories(file_results):
    """Aggregate results by various categories."""
    aggregations = {
        'by_split': defaultdict(list),
        'by_language': defaultdict(list),
        'by_dataset': defaultdict(list),
        'by_voice': defaultdict(list),
        'by_split_language': defaultdict(list),
        'by_split_dataset': defaultdict(list),
        'by_language_dataset': defaultdict(list)
    }
    
    for result in file_results:
        duration = result['audio_duration']
        split = result['split']
        category = result['category']
        language = category['language']
        dataset = category['dataset_type']
        voice = category['voice']
        
        # Single dimension aggregations
        aggregations['by_split'][split].append(duration)
        aggregations['by_language'][language].append(duration)
        aggregations['by_dataset'][dataset].append(duration)
        aggregations['by_voice'][voice].append(duration)
        
        # Multi-dimensional aggregations
        aggregations['by_split_language'][f"{split}_{language}"].append(duration)
        aggregations['by_split_dataset'][f"{split}_{dataset}"].append(duration)
        aggregations['by_language_dataset'][f"{language}_{dataset}"].append(duration)
    
    return aggregations

def analyze_duration_thresholds(lengths):
    """Analyze clip counts at various duration thresholds."""
    thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    analysis = {}
    
    for threshold in thresholds:
        count = sum(1 for length in lengths if length < threshold)
        analysis[f"<{threshold}s"] = count
    
    analysis[">30s"] = sum(1 for length in lengths if length > 30)
    analysis["total"] = len(lengths)
    
    return analysis

def find_file_pairs_from_cache(cached_mapping, split_dir):
    """Use cached mapping to find file pairs without filesystem scanning."""
    file_pairs = []
    split_name = os.path.basename(split_dir)
    
    print(f"Using cached data to find file pairs for {split_name} split...")
    
    for wav_path, file_data in cached_mapping.items():
        if file_data.get('split') == split_name:
            txt_path = file_data.get('txt_path')
            file_pairs.append((wav_path, txt_path))
    
    print(f"Found {len(file_pairs)} cached file pairs for {split_name}")
    return file_pairs

def find_file_pairs_fast(split_dir):
    """Efficiently find wav/text file pairs using pathlib."""
    file_pairs = []
    split_path = Path(split_dir)
    
    # Use glob to find all wav files efficiently
    wav_files = list(split_path.rglob('*.wav'))
    
    print(f"Found {len(wav_files)} WAV files, checking for text pairs...")
    
    # Process in batches for better memory usage and progress reporting
    batch_size = 10000
    for i in range(0, len(wav_files), batch_size):
        batch = wav_files[i:i+batch_size]
        print(f"Checking batch {i//batch_size + 1}/{(len(wav_files)-1)//batch_size + 1}...")
        
        for wav_path in batch:
            base_name = wav_path.stem
            parent_dir = wav_path.parent
            
            # Check for text file (prioritize .normalized.txt)
            normalized_txt = parent_dir / f"{base_name}.normalized.txt"
            regular_txt = parent_dir / f"{base_name}.txt"
            
            try:
                if normalized_txt.exists():
                    file_pairs.append((str(wav_path), str(normalized_txt)))
                elif regular_txt.exists():
                    file_pairs.append((str(wav_path), str(regular_txt)))
                else:
                    # If no text file exists, still add the wav file with None for text
                    # This prevents crashes but marks the text as missing
                    file_pairs.append((str(wav_path), None))
            except OSError as e:
                # Handle cases where filename is too long (common with German EmoNet files)
                if "File name too long" in str(e):
                    print(f"    Warning: Filename too long, skipping: {wav_path.name[:100]}...")
                    print(f"    Note: Consider running the German EmoNet renaming script to fix this issue")
                    continue
                else:
                    # Re-raise other OSErrors
                    raise
    
    print(f"Found {len(file_pairs)} valid file pairs")
    return file_pairs

def summarize_durations(root_dir, sample_size=None, max_workers=None, splits=None, 
                       mapping_file=None, use_cache=True):
    if splits is None:
        splits = ['train', 'dev', 'test']
    
    results = {}
    clip_lengths = {}
    detailed_analysis = {}
    
    # Mapping file setup
    if mapping_file is None:
        mapping_file = os.path.join(os.path.dirname(root_dir), 'audio_duration_mapping.json')
    
    # Load existing mapping if available
    existing_mapping = load_existing_mapping(mapping_file) if use_cache else {}
    print(f"Loaded {len(existing_mapping)} entries from existing mapping file")
    
    # Use optimal number of workers (conservative for I/O heavy tasks)
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)  # Reduced back to 8 for stability
    
    print(f"Using {max_workers} parallel workers")
    if sample_size:
        print(f"Sampling mode: processing max {sample_size} files per split")
    
    # Collect all file results for comprehensive analysis
    all_file_results = []
    new_mapping_entries = {}
    
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: '{split_dir}' not found, skipping.")
            continue

        print(f"\n{'='*50}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*50}")
        
        # Use cached data if available, otherwise scan filesystem
        if existing_mapping:
            file_pairs = find_file_pairs_from_cache(existing_mapping, split_dir)
        else:
            file_pairs = find_file_pairs_fast(split_dir)
        
        if not file_pairs:
            print(f"No valid file pairs found in {split} split.")
            continue
        
        # Apply sampling if requested
        if sample_size and len(file_pairs) > sample_size:
            import random
            random.seed(42)  # For reproducible sampling
            file_pairs = random.sample(file_pairs, sample_size)
            print(f"Sampled {sample_size} files from {len(file_pairs)} total")

        print(f"Processing {len(file_pairs)} file pairs with {max_workers} workers...")
        
        # Prepare file pairs with split info for enhanced processing
        file_pairs_with_split = [(pair, split) for pair in file_pairs]
        
        # Check cache for existing entries - if using cached file pairs, most should be cached
        files_to_process = []
        cached_results = []
        
        for pair_with_split in file_pairs_with_split:
            pair, split_name = pair_with_split
            wav_path, txt_path = pair
            
            # Create file identifier
            file_key = wav_path
            
            # Check if we can use cached result
            if use_cache and file_key in existing_mapping:
                cached_entry = existing_mapping[file_key]
                # If using cached file pairs, trust the cache more
                if existing_mapping:
                    cached_results.append(cached_entry)
                    continue
                else:
                    # Check if file hasn't changed (simple size/mtime check)
                    try:
                        current_size = os.path.getsize(wav_path)
                        current_mtime = os.path.getmtime(wav_path)
                        if (cached_entry.get('file_size_wav') == current_size and 
                            cached_entry.get('last_modified') == current_mtime):
                            cached_results.append(cached_entry)
                            continue
                    except OSError:
                        pass  # File doesn't exist, need to process
            
            files_to_process.append(pair_with_split)
        
        print(f"Using {len(cached_results)} cached results, processing {len(files_to_process)} new files")
        
        # Initialize counters
        total_sec = 0.0
        lengths = []
        empty_wavs = 0
        corrupted_wavs = 0
        missing_texts = 0
        empty_texts = 0
        short_texts = 0
        text_stats = []
        
        # Add cached results to counters
        for cached_result in cached_results:
            dur = cached_result['audio_duration']
            total_sec += dur
            lengths.append(dur)
            all_file_results.append(cached_result)
            
            if cached_result.get('audio_error'):
                corrupted_wavs += 1
            elif dur == 0.0:
                empty_wavs += 1
            elif dur < 0.1:
                corrupted_wavs += 1
            
            text_info = cached_result.get('text_info', {})
            text_stats.append(text_info)
            
            if not text_info.get('exists', False):
                missing_texts += 1
            elif text_info.get('is_empty', False):
                empty_texts += 1
            elif text_info.get('is_too_short', False):
                short_texts += 1
        
        # Process new files in parallel - batch processing for better performance
        if files_to_process:
            batch_size = min(5000, len(files_to_process))  # Process in batches to avoid memory issues
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for i in range(0, len(files_to_process), batch_size):
                    batch = files_to_process[i:i+batch_size]
                    batch_end = min(i + batch_size, len(files_to_process))
                    
                    print(f"Processing batch {i//batch_size + 1}/{(len(files_to_process)-1)//batch_size + 1} "
                          f"(files {i+1}-{batch_end})")
                    
                    # Submit batch
                    futures = [executor.submit(process_file_pair_enhanced, pair_with_split) for pair_with_split in batch]
                    
                    # Process results without tqdm for speed
                    for j, future in enumerate(futures):
                        if j % 1000 == 0:  # Simple progress indicator
                            print(f"  Processed {i + j + 1}/{len(files_to_process)} new files...")
                        
                        try:
                            result = future.result()
                            all_file_results.append(result)
                            new_mapping_entries[result['wav_path']] = result
                            
                            # Process audio results
                            dur = result['audio_duration']
                            total_sec += dur
                            lengths.append(dur)
                            
                            if result['audio_error']:
                                corrupted_wavs += 1
                            elif dur == 0.0:
                                empty_wavs += 1
                            elif dur < 0.1:
                                corrupted_wavs += 1
                            
                            # Process text results
                            text_info = result['text_info']
                            text_stats.append(text_info)
                            
                            if not text_info['exists']:
                                missing_texts += 1
                            elif text_info['is_empty']:
                                empty_texts += 1
                            elif text_info['is_too_short']:
                                short_texts += 1
                                
                        except Exception as e:
                            print(f"  [Processing Error]: {e}")
                            corrupted_wavs += 1
                            missing_texts += 1
                            lengths.append(0.0)

        hours = total_sec / 3600
        results[split] = (total_sec, hours)
        clip_lengths[split] = lengths
        
        # Detailed analysis
        duration_analysis = analyze_duration_thresholds(lengths)
        
        detailed_analysis[split] = {
            'duration_thresholds': duration_analysis,
            'audio_issues': {
                'empty_wavs': empty_wavs,
                'corrupted_wavs': corrupted_wavs,
                'total_problematic_audio': empty_wavs + corrupted_wavs
            },
            'text_issues': {
                'missing_texts': missing_texts,
                'empty_texts': empty_texts,
                'short_texts': short_texts,
                'total_problematic_text': missing_texts + empty_texts + short_texts
            },
            'text_stats': {
                'avg_char_count': np.mean([t['char_count'] for t in text_stats if t['char_count'] > 0]) if text_stats else 0,
                'avg_word_count': np.mean([t['word_count'] for t in text_stats if t['word_count'] > 0]) if text_stats else 0,
                'total_files': len(file_pairs)
            }
        }

    # Save updated mapping file
    if new_mapping_entries:
        updated_mapping = {**existing_mapping, **new_mapping_entries}
        save_mapping_file(updated_mapping, mapping_file)
        print(f"Added {len(new_mapping_entries)} new entries to mapping file")
    else:
        print("No new entries to add to mapping file")
    
    # Generate comprehensive category analysis
    category_analysis = aggregate_by_categories(all_file_results)
    
    return results, clip_lengths, detailed_analysis, category_analysis, all_file_results

def format_timedelta(seconds):
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{td.days*24 + h}h {m}m {s}s"

def print_category_summary(category_data, title):
    """Print summary for a specific category breakdown."""
    print(f"\n{title}")
    print("-" * len(title))
    
    total_overall = 0
    for category, durations in category_data.items():
        if durations:
            total_seconds = sum(durations)
            total_hours = total_seconds / 3600
            total_overall += total_seconds
            count = len(durations)
            avg_duration = total_seconds / count if count > 0 else 0
            print(f"  {category:20} → {total_hours:6.2f}h ({format_timedelta(total_seconds)}) - {count:5} files (avg: {avg_duration:.2f}s)")
    
    if total_overall > 0:
        print(f"  {'TOTAL':20} → {total_overall/3600:6.2f}h ({format_timedelta(total_overall)})")

def print_enhanced_analysis(category_analysis):
    """Print comprehensive category-based analysis."""
    print("\n" + "="*80)
    print("ENHANCED CATEGORY ANALYSIS")
    print("="*80)
    
    # Language breakdown
    print_category_summary(category_analysis['by_language'], "BY LANGUAGE")
    
    # Dataset type breakdown  
    print_category_summary(category_analysis['by_dataset'], "BY DATASET TYPE")
    
    # Split breakdown (original)
    print_category_summary(category_analysis['by_split'], "BY SPLIT")
    
    # Combined breakdowns
    print_category_summary(category_analysis['by_language_dataset'], "BY LANGUAGE + DATASET")
    
    print_category_summary(category_analysis['by_split_language'], "BY SPLIT + LANGUAGE")
    
    print_category_summary(category_analysis['by_split_dataset'], "BY SPLIT + DATASET")
    
    # Voice/Speaker summary (top 20 most common)
    voice_data = category_analysis['by_voice']
    if voice_data:
        print(f"\nTOP VOICES/SPEAKERS (by duration)")
        print("-" * 40)
        voice_hours = {voice: sum(durations)/3600 for voice, durations in voice_data.items()}
        sorted_voices = sorted(voice_hours.items(), key=lambda x: x[1], reverse=True)
        
        for i, (voice, hours) in enumerate(sorted_voices[:20]):
            count = len(voice_data[voice])
            print(f"  {i+1:2}. {voice:20} → {hours:6.2f}h ({count:5} files)")
        
        if len(sorted_voices) > 20:
            remaining_hours = sum(hours for voice, hours in sorted_voices[20:])
            remaining_count = sum(len(voice_data[voice]) for voice, _ in sorted_voices[20:])
            print(f"  ... {len(sorted_voices)-20:2} more voices     → {remaining_hours:6.2f}h ({remaining_count:5} files)")

def generate_chunking_recommendations(category_analysis, target_hours_list=[50, 100, 250, 500, 1000]):
    """Generate recommendations for creating balanced dataset chunks with proper train/dev/test splits."""
    print("\n" + "="*80)
    print("DATASET SPLITTING RECOMMENDATIONS FOR TTS RESEARCH")
    print("="*80)
    
    # Calculate available data by language and dataset
    lang_data = category_analysis['by_language']
    dataset_data = category_analysis['by_dataset']
    
    print("Available data:")
    for lang, durations in lang_data.items():
        if durations:
            hours = sum(durations) / 3600
            print(f"  {lang}: {hours:.1f} hours ({len(durations)} files)")
    
    print()
    for dataset, durations in dataset_data.items():
        if durations:
            hours = sum(durations) / 3600
            print(f"  {dataset}: {hours:.1f} hours ({len(durations)} files)")
    
    fr_hours = sum(lang_data.get('FR', [])) / 3600
    de_hours = sum(lang_data.get('DE', [])) / 3600
    
    print(f"\nRecommended evaluation set sizes (based on TTS research best practices):")
    print(f"  - Dev set: 1.5h per language (3h total, ~1,000-2,000 samples)")
    print(f"  - Test set: 1h per language (2h total, ~500-1,000 samples)")
    print(f"  - This follows typical TTS evaluation practices for rigorous but feasible evaluation")
    
    print(f"\nDataset composition recommendations:")
    print(f"  - Default: 80% LibriSpeech, 20% EmoNet (recommended for stability)")
    print(f"  - Conservative: 85% LibriSpeech, 15% EmoNet (minimal emotional artifacts)")
    print(f"  - Expressive: 70% LibriSpeech, 30% EmoNet (more natural but risk of unwanted emotions)")
    print(f"  - Current data ratio: {fr_hours+de_hours > 0 and (sum(dataset_data.get('LibriSpeech', []))/3600) / ((fr_hours+de_hours)) * 100:.0f}% LibriSpeech, {fr_hours+de_hours > 0 and (sum(dataset_data.get('EmoNet', []))/3600) / ((fr_hours+de_hours)) * 100:.0f}% EmoNet")
    
    print(f"\nTrain set recommendations:")
    print("Format: TARGET → Actual train hours (Total hours including dev/test)")
    print("Note: All splits will be 50% French, 50% German")
    
    for target_train in target_hours_list:
        # Account for dev/test overhead
        dev_test_overhead = 5.0  # 3h dev + 2h test
        total_needed = target_train + dev_test_overhead
        
        train_per_lang = target_train / 2
        needed_per_lang = total_needed / 2
        
        if needed_per_lang <= min(fr_hours, de_hours):
            status = "✓ Feasible"
            fr_pct = (needed_per_lang / fr_hours * 100) if fr_hours > 0 else 0
            de_pct = (needed_per_lang / de_hours * 100) if de_hours > 0 else 0
        else:
            status = "✗ Not feasible with balanced split"
            fr_pct = de_pct = 0
        
        print(f"  {target_train:4}h train → {train_per_lang:.1f}h per lang ({total_needed:.1f}h total): {status}")
        if fr_pct > 0:
            print(f"        Uses {fr_pct:.1f}% of FR data, {de_pct:.1f}% of DE data")
    
    print(f"\nUsage examples:")
    print(f"  # Create splits for 100h training data (default: 80% LibriSpeech, 20% EmoNet)")
    print(f"  python {__file__.split('/')[-1]} your_data_dir --create-splits 100")
    print(f"  ")
    print(f"  # Create conservative 85/15 split (less emotional artifacts)")
    print(f"  python {__file__.split('/')[-1]} your_data_dir --create-splits 100 --libri-ratio 0.85")
    print(f"  ")
    print(f"  # Create multiple splits with custom ratios and dev/test sizes")
    print(f"  python {__file__.split('/')[-1]} your_data_dir --create-splits 50 100 500 --libri-ratio 0.8 --dev-hours 2 --test-hours 1.5")
    print(f"  ")
    print(f"  # The script will create files with dataset ratio in filename:")
    print(f"  #   - sophisticated_balanced_splits_XXXh_train_libriYY.json (detailed metadata)")
    print(f"  #   - train_XXXh_train_libriYY.txt, dev_XXXh_train_libriYY.txt, test_XXXh_train_libriYY.txt")

def create_sophisticated_balanced_splits(all_file_results, target_train_hours, 
                                       dev_hours_per_lang=1.5, test_hours_per_lang=1.0, 
                                       libri_ratio=0.8, output_dir=None):
    """Create sophisticated train/dev/test splits with balanced languages and dataset ratios.
    
    Args:
        all_file_results: All processed file results
        target_train_hours: Target training hours (will be split 50/50 between languages)
        dev_hours_per_lang: Hours per language for dev set (default: 1.5h each = 3h total)
        test_hours_per_lang: Hours per language for test set (default: 1h each = 2h total)
        libri_ratio: Ratio of LibriSpeech to total data (default: 0.8 for 80% LibriSpeech, 20% EmoNet)
        output_dir: Output directory for files
    
    Returns:
        Dictionary with split information and file paths
    """
    if output_dir is None:
        output_dir = "/tmp"
    
    emonet_ratio = 1.0 - libri_ratio
    
    print(f"\n" + "="*80)
    print("CREATING SOPHISTICATED BALANCED SPLITS WITH DATASET RATIOS")
    print("="*80)
    print(f"Target training hours: {target_train_hours}h ({target_train_hours/2:.1f}h per language)")
    print(f"Dataset composition: {libri_ratio*100:.0f}% LibriSpeech, {emonet_ratio*100:.0f}% EmoNet")
    print(f"Dev set: {dev_hours_per_lang}h per language ({dev_hours_per_lang*2:.1f}h total)")
    print(f"Test set: {test_hours_per_lang}h per language ({test_hours_per_lang*2:.1f}h total)")
    print(f"Total target: {target_train_hours + (dev_hours_per_lang + test_hours_per_lang)*2:.1f}h")
    
    # Separate valid files by language AND dataset, exclude corrupted files
    files_by_lang_dataset = {
        'FR': {'LibriSpeech': [], 'EmoNet': []},
        'DE': {'LibriSpeech': [], 'EmoNet': []}
    }
    
    for r in all_file_results:
        if (not r.get('audio_error') and not r['text_info'].get('error') 
            and not r['text_info'].get('is_empty')):
            lang = r['category']['language']
            dataset = r['category']['dataset_type']
            if lang in files_by_lang_dataset and dataset in files_by_lang_dataset[lang]:
                files_by_lang_dataset[lang][dataset].append(r)
    
    print(f"Valid files by language and dataset:")
    for lang in ['FR', 'DE']:
        libri_count = len(files_by_lang_dataset[lang]['LibriSpeech'])
        emonet_count = len(files_by_lang_dataset[lang]['EmoNet'])
        libri_hours = sum(f['audio_duration'] for f in files_by_lang_dataset[lang]['LibriSpeech']) / 3600
        emonet_hours = sum(f['audio_duration'] for f in files_by_lang_dataset[lang]['EmoNet']) / 3600
        print(f"  {lang}: LibriSpeech={libri_count} files ({libri_hours:.1f}h), EmoNet={emonet_count} files ({emonet_hours:.1f}h)")
    
    # Sort files by duration for better distribution
    for lang in files_by_lang_dataset:
        for dataset in files_by_lang_dataset[lang]:
            files_by_lang_dataset[lang][dataset].sort(key=lambda x: x['audio_duration'])
    
    # Convert to seconds
    dev_seconds_per_lang = dev_hours_per_lang * 3600
    test_seconds_per_lang = test_hours_per_lang * 3600
    train_seconds_per_lang = target_train_hours * 3600 / 2
    
    def select_files_with_dataset_ratio(lang_files, target_seconds, split_name, lang):
        """Select files ensuring both good duration distribution AND dataset ratio."""
        libri_files = lang_files['LibriSpeech'].copy()
        emonet_files = lang_files['EmoNet'].copy()
        
        # Calculate target seconds per dataset
        libri_target = target_seconds * libri_ratio
        emonet_target = target_seconds * emonet_ratio
        
        print(f"  {lang} {split_name} targets: LibriSpeech={libri_target/3600:.2f}h, EmoNet={emonet_target/3600:.2f}h")
        
        selected = {'LibriSpeech': [], 'EmoNet': []}
        selected_indices = {'LibriSpeech': set(), 'EmoNet': set()}  # Track indices for fast removal
        totals = {'LibriSpeech': 0, 'EmoNet': 0}
        
        # Select files from each dataset to meet ratio targets
        for dataset_name, files, target in [('LibriSpeech', libri_files, libri_target), 
                                          ('EmoNet', emonet_files, emonet_target)]:
            if not files:
                print(f"    Warning: No {dataset_name} files available for {lang}")
                continue
                
            # Use simple greedy selection for speed - just iterate through sorted files
            current_total = 0
            for idx, file_result in enumerate(files):
                if current_total >= target:
                    break
                selected[dataset_name].append(file_result)
                selected_indices[dataset_name].add(idx)
                current_total += file_result['audio_duration']
            
            totals[dataset_name] = current_total
        
        # Combine selected files from both datasets
        all_selected = selected['LibriSpeech'] + selected['EmoNet']
        total_seconds = sum(f['audio_duration'] for f in all_selected)
        
        # Report actual ratios achieved
        if total_seconds > 0:
            actual_libri_ratio = totals['LibriSpeech'] / total_seconds
            actual_emonet_ratio = totals['EmoNet'] / total_seconds
            print(f"    {lang} {split_name} actual: LibriSpeech={actual_libri_ratio:.1%} ({totals['LibriSpeech']/3600:.2f}h), "
                  f"EmoNet={actual_emonet_ratio:.1%} ({totals['EmoNet']/3600:.2f}h)")
        
        return all_selected, total_seconds, selected, selected_indices
    
    # Create splits for each language with dataset ratios
    splits_data = {'fr': {}, 'de': {}}
    
    for lang_key, lang_display in [('fr', 'FR'), ('de', 'DE')]:
        lang_files = files_by_lang_dataset[lang_display]
        
        # First, reserve files for test (best quality, diverse durations, respect dataset ratio)
        test_files, test_total, test_by_dataset, test_indices = select_files_with_dataset_ratio(
            lang_files, test_seconds_per_lang, 'test', lang_display)
        
        # Remove selected test files from remaining pools using efficient index-based removal
        for dataset in ['LibriSpeech', 'EmoNet']:
            if test_indices[dataset]:
                # Sort indices in reverse order to remove from end first (maintains other indices)
                sorted_indices = sorted(test_indices[dataset], reverse=True)
                for idx in sorted_indices:
                    lang_files[dataset].pop(idx)
        
        # Then, reserve files for dev
        dev_files, dev_total, dev_by_dataset, dev_indices = select_files_with_dataset_ratio(
            lang_files, dev_seconds_per_lang, 'dev', lang_display)
        
        # Remove selected dev files from remaining pools
        for dataset in ['LibriSpeech', 'EmoNet']:
            if dev_indices[dataset]:
                sorted_indices = sorted(dev_indices[dataset], reverse=True)
                for idx in sorted_indices:
                    lang_files[dataset].pop(idx)
        
        # Finally, select train files
        train_files, train_total, train_by_dataset, _ = select_files_with_dataset_ratio(
            lang_files, train_seconds_per_lang, 'train', lang_display)
        
        splits_data[lang_key] = {
            'train': {'files': train_files, 'seconds': train_total, 'by_dataset': train_by_dataset},
            'dev': {'files': dev_files, 'seconds': dev_total, 'by_dataset': dev_by_dataset},
            'test': {'files': test_files, 'seconds': test_total, 'by_dataset': test_by_dataset}
        }
        
        print(f"\n{lang_display} splits:")
        for split in ['train', 'dev', 'test']:
            split_data = splits_data[lang_key][split]
            libri_count = len(split_data['by_dataset']['LibriSpeech'])
            emonet_count = len(split_data['by_dataset']['EmoNet'])
            libri_hours = sum(f['audio_duration'] for f in split_data['by_dataset']['LibriSpeech']) / 3600
            emonet_hours = sum(f['audio_duration'] for f in split_data['by_dataset']['EmoNet']) / 3600
            total_hours = split_data['seconds'] / 3600
            print(f"  {split.capitalize()}: {total_hours:.2f}h ({len(split_data['files'])} files)")
            print(f"    LibriSpeech: {libri_hours:.2f}h ({libri_count} files)")
            print(f"    EmoNet: {emonet_hours:.2f}h ({emonet_count} files)")
    
    # Combine splits and save
    output_file = os.path.join(output_dir, f'sophisticated_balanced_splits_{target_train_hours}h_train_libri{int(libri_ratio*100)}.json')
    
    # Prepare summary statistics
    total_train_hours = (splits_data['fr']['train']['seconds'] + splits_data['de']['train']['seconds']) / 3600
    total_dev_hours = (splits_data['fr']['dev']['seconds'] + splits_data['de']['dev']['seconds']) / 3600
    total_test_hours = (splits_data['fr']['test']['seconds'] + splits_data['de']['test']['seconds']) / 3600
    
    # Calculate actual dataset ratios across all splits
    def calculate_dataset_totals():
        totals = {'LibriSpeech': 0, 'EmoNet': 0}
        for lang_key in ['fr', 'de']:
            for split in ['train', 'dev', 'test']:
                for dataset in ['LibriSpeech', 'EmoNet']:
                    dataset_seconds = sum(f['audio_duration'] for f in splits_data[lang_key][split]['by_dataset'][dataset])
                    totals[dataset] += dataset_seconds
        return totals
    
    dataset_totals = calculate_dataset_totals()
    total_seconds = sum(dataset_totals.values())
    actual_libri_ratio = dataset_totals['LibriSpeech'] / total_seconds if total_seconds > 0 else 0
    actual_emonet_ratio = dataset_totals['EmoNet'] / total_seconds if total_seconds > 0 else 0
    
    sample_data = {
        'metadata': {
            'target_train_hours': target_train_hours,
            'actual_train_hours': total_train_hours,
            'dev_hours_per_language': dev_hours_per_lang,
            'test_hours_per_language': test_hours_per_lang,
            'actual_total_hours': total_train_hours + total_dev_hours + total_test_hours,
            'target_libri_ratio': libri_ratio,
            'target_emonet_ratio': emonet_ratio,
            'actual_libri_ratio': actual_libri_ratio,
            'actual_emonet_ratio': actual_emonet_ratio,
            'creation_date': str(datetime.datetime.now()),
        },
        'dataset_composition': {
            'total_libri_hours': dataset_totals['LibriSpeech'] / 3600,
            'total_emonet_hours': dataset_totals['EmoNet'] / 3600,
            'actual_libri_percentage': actual_libri_ratio * 100,
            'actual_emonet_percentage': actual_emonet_ratio * 100
        },
        'summary': {
            'train': {
                'total_hours': total_train_hours,
                'fr_hours': splits_data['fr']['train']['seconds'] / 3600,
                'de_hours': splits_data['de']['train']['seconds'] / 3600,
                'fr_files': len(splits_data['fr']['train']['files']),
                'de_files': len(splits_data['de']['train']['files']),
                'total_files': len(splits_data['fr']['train']['files']) + len(splits_data['de']['train']['files']),
                'libri_hours': sum(f['audio_duration'] for f in splits_data['fr']['train']['by_dataset']['LibriSpeech'] + splits_data['de']['train']['by_dataset']['LibriSpeech']) / 3600,
                'emonet_hours': sum(f['audio_duration'] for f in splits_data['fr']['train']['by_dataset']['EmoNet'] + splits_data['de']['train']['by_dataset']['EmoNet']) / 3600
            },
            'dev': {
                'total_hours': total_dev_hours,
                'fr_hours': splits_data['fr']['dev']['seconds'] / 3600,
                'de_hours': splits_data['de']['dev']['seconds'] / 3600,
                'fr_files': len(splits_data['fr']['dev']['files']),
                'de_files': len(splits_data['de']['dev']['files']),
                'total_files': len(splits_data['fr']['dev']['files']) + len(splits_data['de']['dev']['files']),
                'libri_hours': sum(f['audio_duration'] for f in splits_data['fr']['dev']['by_dataset']['LibriSpeech'] + splits_data['de']['dev']['by_dataset']['LibriSpeech']) / 3600,
                'emonet_hours': sum(f['audio_duration'] for f in splits_data['fr']['dev']['by_dataset']['EmoNet'] + splits_data['de']['dev']['by_dataset']['EmoNet']) / 3600
            },
            'test': {
                'total_hours': total_test_hours,
                'fr_hours': splits_data['fr']['test']['seconds'] / 3600,
                'de_hours': splits_data['de']['test']['seconds'] / 3600,
                'fr_files': len(splits_data['fr']['test']['files']),
                'de_files': len(splits_data['de']['test']['files']),
                'total_files': len(splits_data['fr']['test']['files']) + len(splits_data['de']['test']['files']),
                'libri_hours': sum(f['audio_duration'] for f in splits_data['fr']['test']['by_dataset']['LibriSpeech'] + splits_data['de']['test']['by_dataset']['LibriSpeech']) / 3600,
                'emonet_hours': sum(f['audio_duration'] for f in splits_data['fr']['test']['by_dataset']['EmoNet'] + splits_data['de']['test']['by_dataset']['EmoNet']) / 3600
            }
        },
        'splits': {}
    }
    
    # Add detailed file lists for each split
    for split in ['train', 'dev', 'test']:
        sample_data['splits'][split] = {
            'fr_files': [{'wav_path': f['wav_path'], 'txt_path': f['txt_path'], 
                         'duration': f['audio_duration'], 'original_split': f['split'],
                         'dataset_type': f['category']['dataset_type'],
                         'voice': f['category']['voice']} 
                        for f in splits_data['fr'][split]['files']],
            'de_files': [{'wav_path': f['wav_path'], 'txt_path': f['txt_path'],
                         'duration': f['audio_duration'], 'original_split': f['split'],
                         'dataset_type': f['category']['dataset_type'], 
                         'voice': f['category']['voice']} 
                        for f in splits_data['de'][split]['files']]
        }
    
    # Save the comprehensive split data
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Also create simple file lists for each split (easier to use)
    for split in ['train', 'dev', 'test']:
        split_file = os.path.join(output_dir, f'{split}_{target_train_hours}h_train_libri{int(libri_ratio*100)}.txt')
        with open(split_file, 'w') as f:
            for file_info in sample_data['splits'][split]['fr_files']:
                f.write(f"{file_info['wav_path']}|{file_info['txt_path']}\n")
            for file_info in sample_data['splits'][split]['de_files']:
                f.write(f"{file_info['wav_path']}|{file_info['txt_path']}\n")
    
    print(f"\nCreated sophisticated balanced splits with dataset ratios:")
    print(f"  Main file: {output_file}")
    print(f"  Simple lists: {os.path.join(output_dir, f'train_{target_train_hours}h_train_libri{int(libri_ratio*100)}.txt')}, etc.")
    print(f"\nActual results:")
    print(f"  Dataset composition: {actual_libri_ratio:.1%} LibriSpeech, {actual_emonet_ratio:.1%} EmoNet")
    print(f"  Train: {total_train_hours:.1f}h ({sample_data['summary']['train']['total_files']} files)")
    print(f"    - FR: {sample_data['summary']['train']['fr_hours']:.1f}h ({sample_data['summary']['train']['fr_files']} files)")
    print(f"    - DE: {sample_data['summary']['train']['de_hours']:.1f}h ({sample_data['summary']['train']['de_files']} files)")
    print(f"    - LibriSpeech: {sample_data['summary']['train']['libri_hours']:.1f}h")
    print(f"    - EmoNet: {sample_data['summary']['train']['emonet_hours']:.1f}h")
    print(f"  Dev: {total_dev_hours:.1f}h ({sample_data['summary']['dev']['total_files']} files)")
    print(f"    - FR: {sample_data['summary']['dev']['fr_hours']:.1f}h ({sample_data['summary']['dev']['fr_files']} files)")
    print(f"    - DE: {sample_data['summary']['dev']['de_hours']:.1f}h ({sample_data['summary']['dev']['de_files']} files)")
    print(f"    - LibriSpeech: {sample_data['summary']['dev']['libri_hours']:.1f}h")
    print(f"    - EmoNet: {sample_data['summary']['dev']['emonet_hours']:.1f}h")
    print(f"  Test: {total_test_hours:.1f}h ({sample_data['summary']['test']['total_files']} files)")
    print(f"    - FR: {sample_data['summary']['test']['fr_hours']:.1f}h ({sample_data['summary']['test']['fr_files']} files)")
    print(f"    - DE: {sample_data['summary']['test']['de_hours']:.1f}h ({sample_data['summary']['test']['de_files']} files)")
    print(f"    - LibriSpeech: {sample_data['summary']['test']['libri_hours']:.1f}h")
    print(f"    - EmoNet: {sample_data['summary']['test']['emonet_hours']:.1f}h")
    
    return {
        'main_file': output_file,
        'split_files': {
            'train': os.path.join(output_dir, f'train_{target_train_hours}h_train_libri{int(libri_ratio*100)}.txt'),
            'dev': os.path.join(output_dir, f'dev_{target_train_hours}h_train_libri{int(libri_ratio*100)}.txt'),
            'test': os.path.join(output_dir, f'test_{target_train_hours}h_train_libri{int(libri_ratio*100)}.txt')
        },
        'summary': sample_data['summary'],
        'dataset_composition': sample_data['dataset_composition']
    }

def create_balanced_sample_lists(all_file_results, target_hours, output_dir=None):
    """Create simple file lists for balanced dataset chunks (legacy function)."""
    if output_dir is None:
        output_dir = "/tmp"
    
    # Separate by language
    fr_files = [r for r in all_file_results if r['category']['language'] == 'FR' and not r.get('audio_error')]
    de_files = [r for r in all_file_results if r['category']['language'] == 'DE' and not r.get('audio_error')]
    
    # Sort by duration for better distribution
    fr_files.sort(key=lambda x: x['audio_duration'])
    de_files.sort(key=lambda x: x['audio_duration'])
    
    target_seconds_per_lang = target_hours * 3600 / 2
    
    # Select files for each language
    selected_fr = []
    selected_de = []
    fr_total = 0
    de_total = 0
    
    # Greedy selection (could be improved with better algorithms)
    for file_result in fr_files:
        if fr_total < target_seconds_per_lang:
            selected_fr.append(file_result)
            fr_total += file_result['audio_duration']
    
    for file_result in de_files:
        if de_total < target_seconds_per_lang:
            selected_de.append(file_result)
            de_total += file_result['audio_duration']
    
    # Save file lists
    output_file = os.path.join(output_dir, f'balanced_dataset_{target_hours}h.json')
    sample_data = {
        'target_hours': target_hours,
        'actual_hours_fr': fr_total / 3600,
        'actual_hours_de': de_total / 3600,
        'actual_hours_total': (fr_total + de_total) / 3600,
        'files_fr': len(selected_fr),
        'files_de': len(selected_de),
        'files_total': len(selected_fr) + len(selected_de),
        'fr_files': [{'wav_path': f['wav_path'], 'duration': f['audio_duration'], 'split': f['split']} for f in selected_fr],
        'de_files': [{'wav_path': f['wav_path'], 'duration': f['audio_duration'], 'split': f['split']} for f in selected_de]
    }
    
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Created balanced sample list: {output_file}")
    print(f"  Target: {target_hours}h, Actual: {sample_data['actual_hours_total']:.1f}h")
    print(f"  FR: {sample_data['actual_hours_fr']:.1f}h ({sample_data['files_fr']} files)")
    print(f"  DE: {sample_data['actual_hours_de']:.1f}h ({sample_data['files_de']} files)")
    
    return output_file
def main():
    parser = argparse.ArgumentParser(
        description="Compute total audio duration in hours for train/dev/test splits and show detailed analysis."
    )
    parser.add_argument(
        'root_dir',
        help="Path to the directory containing 'train', 'dev', 'test' subfolders"
    )
    parser.add_argument(
        '--sample-size', type=int, default=None,
        help="Process only a sample of files for quick testing (e.g., --sample-size 1000)"
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help="Number of parallel workers (default: min(CPU_count, 8))"
    )
    parser.add_argument(
        '--splits', nargs='+', default=['train', 'dev', 'test'],
        help="Which splits to process (default: train dev test)"
    )
    parser.add_argument(
        '--mapping-file', type=str, default=None,
        help="Path to save/load the comprehensive mapping file (default: auto-generate)"
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help="Disable caching and reprocess all files"
    )
    parser.add_argument(
        '--create-splits', nargs='+', type=int, default=None,
        help="Create sophisticated train/dev/test splits for specified train hours (e.g., --create-splits 50 100 250)"
    )
    parser.add_argument(
        '--dev-hours', type=float, default=1.5,
        help="Hours per language for dev set (default: 1.5h per language = 3h total)"
    )
    parser.add_argument(
        '--test-hours', type=float, default=1.0,
        help="Hours per language for test set (default: 1h per language = 2h total)"
    )
    parser.add_argument(
        '--libri-ratio', type=float, default=0.8,
        help="Ratio of LibriSpeech to total data (default: 0.8 for 80%% LibriSpeech, 20%% EmoNet)"
    )
    parser.add_argument(
        '--create-samples', nargs='+', type=int, default=None,
        help="Create simple balanced sample file lists for specified hours (e.g., --create-samples 50 100 250) - LEGACY"
    )
    args = parser.parse_args()

    res, clip_lengths, detailed_analysis, category_analysis, all_file_results = summarize_durations(
        args.root_dir, 
        sample_size=args.sample_size,
        max_workers=args.workers,
        splits=args.splits,
        mapping_file=args.mapping_file,
        use_cache=not args.no_cache
    )
    
    # Original summary
    print("\n" + "="*60)
    print("AUDIO DURATION SUMMARY")
    print("="*60)
    for split, (secs, hrs) in res.items():
        print(f"  {split:5} → {hrs:.2f} hours ({format_timedelta(secs)})")

    # Enhanced category analysis
    print_enhanced_analysis(category_analysis)
    
    # Chunking recommendations
    generate_chunking_recommendations(category_analysis)

    print("\n" + "="*60)
    print("DETAILED DURATION ANALYSIS (BY SPLIT)")
    print("="*60)
    
    for split in detailed_analysis:
        print(f"\n{split.upper()} SPLIT:")
        print("-" * 20)
        
        # Duration thresholds
        duration_data = detailed_analysis[split]['duration_thresholds']
        total_files = duration_data['total']
        
        print("Duration thresholds:")
        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
        for threshold in thresholds:
            count = duration_data[f'<{threshold}s']
            percentage = (count / total_files * 100) if total_files > 0 else 0
            print(f"  <{threshold:2}s: {count:6} files ({percentage:5.1f}%)")
        
        over_30_count = duration_data['>30s']
        over_30_pct = (over_30_count / total_files * 100) if total_files > 0 else 0
        print(f"  >30s: {over_30_count:6} files ({over_30_pct:5.1f}%)")
        
        # Audio issues
        audio_issues = detailed_analysis[split]['audio_issues']
        print(f"\nAudio file issues:")
        print(f"  Empty/corrupted wavs: {audio_issues['empty_wavs']} empty, {audio_issues['corrupted_wavs']} corrupted")
        print(f"  Total problematic audio: {audio_issues['total_problematic_audio']}")
        
        # Text issues
        text_issues = detailed_analysis[split]['text_issues']
        text_stats = detailed_analysis[split]['text_stats']
        print(f"\nText file issues:")
        print(f"  Missing text files: {text_issues['missing_texts']}")
        print(f"  Empty text files: {text_issues['empty_texts']}")
        print(f"  Very short texts (<3 chars): {text_issues['short_texts']}")
        print(f"  Total problematic text: {text_issues['total_problematic_text']}")
        
        print(f"\nText statistics:")
        print(f"  Average characters per text: {text_stats['avg_char_count']:.1f}")
        print(f"  Average words per text: {text_stats['avg_word_count']:.1f}")
        print(f"  Total file pairs: {text_stats['total_files']}")

    print("\n" + "="*60)
    print("FILTERING RECOMMENDATIONS")
    print("="*60)
    
    for split in detailed_analysis:
        duration_data = detailed_analysis[split]['duration_thresholds']
        audio_issues = detailed_analysis[split]['audio_issues']
        text_issues = detailed_analysis[split]['text_issues']
        total_files = duration_data['total']
        
        # Calculate potential filtering impact
        very_short = duration_data['<1s']
        very_long = duration_data['>30s']
        problematic_audio = audio_issues['total_problematic_audio']
        problematic_text = text_issues['total_problematic_text']
        
        # Estimate overlap (conservative)
        total_to_filter = very_short + very_long + problematic_audio + problematic_text
        # Assume some overlap, so reduce by 20%
        estimated_filtered = int(total_to_filter * 0.8)
        remaining_files = total_files - estimated_filtered
        remaining_pct = (remaining_files / total_files * 100) if total_files > 0 else 0
        
        print(f"\n{split.upper()}:")
        print(f"  Suggested filters:")
        print(f"    - Remove clips <1s: {very_short} files")
        print(f"    - Remove clips >30s: {very_long} files")
        print(f"    - Remove corrupted audio: {problematic_audio} files")
        print(f"    - Remove missing/empty/short text: {problematic_text} files")
        print(f"  Estimated files after filtering: {remaining_files}/{total_files} ({remaining_pct:.1f}%)")

    # Create sophisticated splits if requested
    if args.create_splits:
        print(f"\n" + "="*80)
        print("CREATING SOPHISTICATED TRAIN/DEV/TEST SPLITS")
        print("="*80)
        print(f"Dataset ratio: {args.libri_ratio:.0%} LibriSpeech, {1-args.libri_ratio:.0%} EmoNet")
        print(f"Dev set: {args.dev_hours}h per language ({args.dev_hours * 2:.1f}h total)")
        print(f"Test set: {args.test_hours}h per language ({args.test_hours * 2:.1f}h total)")
        
        output_dir = os.path.dirname(args.root_dir)
        for target_train_hours in args.create_splits:
            split_info = create_sophisticated_balanced_splits(
                all_file_results, target_train_hours, 
                dev_hours_per_lang=args.dev_hours,
                test_hours_per_lang=args.test_hours,
                libri_ratio=args.libri_ratio,
                output_dir=output_dir
            )
            print(f"\nCreated splits for {target_train_hours}h training data:")
            print(f"  Files: {split_info['main_file']}")
            dataset_comp = split_info['dataset_composition']
            print(f"  Dataset composition: {dataset_comp['actual_libri_percentage']:.1f}% LibriSpeech ({dataset_comp['total_libri_hours']:.1f}h), "
                  f"{dataset_comp['actual_emonet_percentage']:.1f}% EmoNet ({dataset_comp['total_emonet_hours']:.1f}h)")
            for split_name, split_file in split_info['split_files'].items():
                summary = split_info['summary'][split_name]
                print(f"  {split_name.capitalize()}: {summary['total_hours']:.1f}h ({summary['total_files']} files) -> {split_file}")
                print(f"    LibriSpeech: {summary['libri_hours']:.1f}h, EmoNet: {summary['emonet_hours']:.1f}h")

    # Create simple balanced samples if requested (legacy functionality)
    if args.create_samples:
        print(f"\n" + "="*60)
        print("CREATING SIMPLE BALANCED SAMPLE LISTS (LEGACY)")
        print("="*60)
        output_dir = os.path.dirname(args.root_dir)
        for target_hours in args.create_samples:
            create_balanced_sample_lists(all_file_results, target_hours, output_dir)

    # Plot histogram if we have data
    if any(clip_lengths.values()) and all_file_results and MATPLOTLIB_AVAILABLE:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # Simple histogram by split
            max_length = max([max(l) if l else 0 for l in clip_lengths.values()])
            bins = np.linspace(0, min(max_length, 60), 60)
            
            for split, lengths in clip_lengths.items():
                if lengths:
                    filtered_lengths = [l for l in lengths if l <= 60]
                    ax.hist(filtered_lengths, bins=bins, alpha=0.6, label=f'{split} ({len(filtered_lengths)} files)')
            
            ax.set_xlabel('Clip length (seconds)')
            ax.set_ylabel('Count')
            ax.set_title('Audio Duration Distribution by Split (0-60s)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('clip_duration_analysis.png', dpi=150, bbox_inches='tight')
            print(f"\nSaved duration analysis plot to 'clip_duration_analysis.png'")
            
        except Exception as e:
            print(f"\nError generating plot: {e}")
    else:
        if not MATPLOTLIB_AVAILABLE:
            print(f"\nMatplotlib not available, skipping plot generation.")
        else:
            print(f"\nNo audio data found for plotting.")

if __name__ == '__main__':
    main()
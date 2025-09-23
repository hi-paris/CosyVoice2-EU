import os
import json
import tarfile
import librosa
import soundfile as sf
import shutil
import random
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_audio_file_batch(args):
    """Process multiple audio files in a batch for better efficiency."""
    extract_dir, audio_files, speaker, file_prefix = args
    
    processed_utts = []
    errors = []
    
    for audio_file in audio_files:
        try:
            audio_path = os.path.join(extract_dir, audio_file)
            json_path = os.path.join(extract_dir, audio_file.replace('.mp3', '.json'))
            
            if not os.path.exists(json_path):
                errors.append(f"Missing JSON for {audio_file}")
                continue
            
            # Extract text from JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            text = data.get("annotation")
            if not text:
                errors.append(f"No annotation in {json_path}")
                continue
                
            # Extract transcription
            if "<transcription_start>" not in text or "</transcription" not in text:
                errors.append(f"Malformed transcription in {json_path}")
                continue
                
            text = text.split("<transcription_start>")[1].split("</transcription")[0].split("<transcription")[0].strip()
            text = text.replace(".\n", ". ").replace(". \n", ". ")
            
            if "\n" in text:
                errors.append(f"Contains newlines in {json_path}")
                continue
            
            # Convert audio efficiently
            base_name = audio_file.replace('.mp3', '')
            utt_id = f"{speaker}_{file_prefix}_{base_name}"
            
            wav_path = os.path.join(extract_dir, f"{utt_id}.wav")
            txt_path = os.path.join(extract_dir, f"{utt_id}.normalized.txt")
            
            # Use librosa with sr=16000 directly for efficiency
            audio, sr = librosa.load(audio_path, sr=16000)  # Most TTS models expect 16kHz
            sf.write(wav_path, audio, sr)
            
            # Write text file
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Clean up original files immediately
            os.remove(audio_path)
            os.remove(json_path)
            
            processed_utts.append({
                'utt_id': utt_id,
                'wav_path': wav_path,
                'txt_path': txt_path,
                'speaker': speaker,
                'subset': file_prefix
            })
            
        except Exception as e:
            errors.append(f"Error processing {audio_file}: {e}")
            continue
    
    return processed_utts, errors


def download_tar_file(args):
    """Download and extract a single tar file efficiently."""
    repo_id, file_path, temp_dir = args
    
    try:
        # Download with explicit cache usage
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset", 
            filename=file_path,
            cache_dir=os.path.join(temp_dir, "hf_cache"),  # Use explicit cache
            local_dir_use_symlinks=False,
        )
        
        # Extract directly to final location to avoid extra moves
        extract_dir = os.path.join(temp_dir, file_path[:-4])
        
        with tarfile.open(local_path, "r") as tar:
            tar.extractall(path=temp_dir)
        
        # Keep the tar file cached but remove our copy
        if os.path.exists(local_path) and local_path.startswith(temp_dir):
            os.remove(local_path)
        
        return extract_dir, None
        
    except Exception as e:
        return None, f"Failed to download {file_path}: {e}"


def check_existing_temp_data(temp_dir):
    """Check if temp data already exists and return the extracted directories."""
    if not os.path.exists(temp_dir):
        return []
    
    downloaded_dirs = []
    for item in os.listdir(temp_dir):
        item_path = os.path.join(temp_dir, item)
        if os.path.isdir(item_path) and item.startswith("fr_") and item != "hf_cache":
            # Extract speaker and file_prefix from directory name
            try:
                speaker = item.split("_")[1]
                file_prefix = item.split("fr_")[1]
                downloaded_dirs.append((item_path, speaker, file_prefix))
            except IndexError:
                continue
    
    logger.info(f"Found {len(downloaded_dirs)} existing extracted directories in temp folder")
    return downloaded_dirs


def load_existing_processed_utterances(temp_dir):
    """Load already processed utterances from temp directories."""
    all_utterances = []
    
    for item in os.listdir(temp_dir):
        item_path = os.path.join(temp_dir, item)
        if os.path.isdir(item_path) and item.startswith("fr_") and item != "hf_cache":
            try:
                speaker = item.split("_")[1]
                file_prefix = item.split("fr_")[1]
                
                # Look for already processed .wav files
                wav_files = [f for f in os.listdir(item_path) if f.endswith('.wav')]
                
                for wav_file in wav_files:
                    txt_file = wav_file.replace('.wav', '.normalized.txt')
                    wav_path = os.path.join(item_path, wav_file)
                    txt_path = os.path.join(item_path, txt_file)
                    
                    if os.path.exists(txt_path):
                        utt_id = wav_file.replace('.wav', '')
                        all_utterances.append({
                            'utt_id': utt_id,
                            'wav_path': wav_path,
                            'txt_path': txt_path,
                            'speaker': speaker,
                            'subset': file_prefix
                        })
                        
            except Exception as e:
                logger.warning(f"Error loading from {item}: {e}")
                continue
    
    return all_utterances


def copy_files_batch(args):
    """Copy multiple files in a batch for better efficiency."""
    utt_list, output_root, split = args
    
    copied_count = 0
    errors = []
    
    for utt_info in utt_list:
        if not os.path.exists(utt_info['wav_path']) or not os.path.exists(utt_info['txt_path']):
            errors.append(f"Missing files for {utt_info['utt_id']}")
            continue
        
        # Create unique filename: speaker_subset_utt.wav
        base_name = f"{utt_info['speaker']}_{utt_info['subset']}_{utt_info['utt_id']}"
        
        dst_wav = os.path.join(output_root, split, base_name + ".wav")
        dst_txt = os.path.join(output_root, split, base_name + ".normalized.txt")
        
        # Skip if both files already exist (optimization for resume)
        if os.path.exists(dst_wav) and os.path.exists(dst_txt):
            continue
        
        try:
            shutil.copy2(utt_info['wav_path'], dst_wav)
            shutil.copy2(utt_info['txt_path'], dst_txt)
            copied_count += 1
        except Exception as e:
            errors.append(f"Failed to copy {utt_info['utt_id']}: {e}")
    
    return copied_count, errors


def download_process_and_split_emonet(data_dir, splits=None, skip_download=False):
    """Download, process, and split EmoNet dataset in one optimized workflow."""
    if splits is None:
        splits = {"train": 0.9, "dev": 0.05, "test": 0.05}
    
    # Ensure reproducibility
    random.seed(42)
    
    repo_id = "laion/laions_got_talent_enhanced_flash_annotations_and_long_captions"
    
    # Create directories
    temp_dir = os.path.join(data_dir, "temp_emonet_optimized")
    output_root = os.path.join(data_dir, "EmoNet_split")
    
    os.makedirs(temp_dir, exist_ok=True)
    for split in splits:
        os.makedirs(os.path.join(output_root, split), exist_ok=True)
    
    # Check for existing temp data first
    downloaded_dirs = check_existing_temp_data(temp_dir)
    
    if skip_download and downloaded_dirs:
        logger.info(f"Skipping download phase - using {len(downloaded_dirs)} existing directories")
    elif downloaded_dirs and not skip_download:
        logger.info(f"Found existing temp data. Set skip_download=True to use it, or continuing with fresh download...")
        # Clear existing for fresh start if not skipping
        for extract_dir, _, _ in downloaded_dirs:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
        downloaded_dirs = []
    
    if not skip_download or not downloaded_dirs:
        logger.info(f"Listing files in repo: {repo_id}...")
        files = list_repo_files(repo_id, repo_type="dataset")
        
        # Filter French files
        french_files = [f for f in files if f.startswith("fr_")]
        logger.info(f"Found {len(french_files)} French files to process")
        
        # Phase 1: Parallel download of tar files
        logger.info("Phase 1: Downloading tar files in parallel...")
        download_args = [(repo_id, file_path, temp_dir) for file_path in french_files]
        
        downloaded_dirs = []
        max_download_workers = min(10, len(french_files))  # Reasonable parallel downloads
        
        with ThreadPoolExecutor(max_workers=max_download_workers) as executor:
            future_to_file = {executor.submit(download_tar_file, args): args[1] for args in download_args}
            
            for future in tqdm(as_completed(future_to_file), total=len(download_args), 
                              desc="Downloading tar files"):
                file_path = future_to_file[future]
                extract_dir, error = future.result()
                
                if error:
                    logger.warning(error)
                    continue
                
                if extract_dir and os.path.exists(extract_dir):
                    speaker = file_path.split("_")[1]
                    file_prefix = file_path.split("fr_")[1].split(".tar")[0]
                    downloaded_dirs.append((extract_dir, speaker, file_prefix))
        
        logger.info(f"Downloaded {len(downloaded_dirs)} tar files successfully")
    
    # Phase 2: Parallel audio processing
    logger.info("Phase 2: Processing audio files in parallel...")
    
    # First, check if we have already processed files
    existing_utterances = load_existing_processed_utterances(temp_dir)
    
    if existing_utterances:
        logger.info(f"Found {len(existing_utterances)} already processed utterances, using them")
        all_utterances = existing_utterances
    else:
        logger.info("No processed utterances found, processing from MP3 files...")
        
        all_utterances = []
        max_audio_workers = min(multiprocessing.cpu_count(), 32)  # Use more CPU cores
        
        # Prepare batches for processing
        processing_tasks = []
        for extract_dir, speaker, file_prefix in downloaded_dirs:
            if not os.path.exists(extract_dir):
                continue
                
            audio_files = [f for f in os.listdir(extract_dir) if f.endswith('.mp3')]
            if not audio_files:
                continue
            
            # Process in batches for efficiency
            batch_size = max(1, len(audio_files) // 4)  # Split into 4 batches per directory
            for i in range(0, len(audio_files), batch_size):
                batch = audio_files[i:i + batch_size]
                processing_tasks.append((extract_dir, batch, speaker, file_prefix))
        
        if processing_tasks:
            # Process audio batches in parallel
            with ProcessPoolExecutor(max_workers=max_audio_workers) as executor:
                future_to_task = {executor.submit(process_audio_file_batch, task): task for task in processing_tasks}
                
                for future in tqdm(as_completed(future_to_task), total=len(processing_tasks),
                                  desc="Processing audio files"):
                    task = future_to_task[future]
                    _, _, speaker, file_prefix = task
                    
                    try:
                        processed_utts, errors = future.result()
                        all_utterances.extend(processed_utts)
                        
                        if errors and len(errors) < 10:  # Log some errors
                            for error in errors[:3]:
                                logger.warning(f"Speaker {speaker}: {error}")
                                
                    except Exception as e:
                        logger.error(f"Failed to process batch for {speaker}: {e}")
        else:
            logger.warning("No MP3 files found to process!")
    
    logger.info(f"Total utterances processed: {len(all_utterances)}")
    
    if len(all_utterances) == 0:
        logger.error("No utterances were processed! This could mean:")
        logger.error("  - All MP3 files were already processed and removed")
        logger.error("  - All JSON files have formatting issues")
        logger.error("  - The temp directory structure is different than expected")
        logger.error("Please check your temp directory manually.")
        return 0
    
    # Phase 3: Split and organize data
    logger.info("Phase 3: Splitting and organizing data...")
    
    # Shuffle all utterances for random split
    random.shuffle(all_utterances)
    
    # Compute split sizes
    n = len(all_utterances)
    n_train = int(n * splits["train"])
    n_dev = int(n * splits["dev"])
    
    split_map = {
        "train": all_utterances[:n_train],
        "dev": all_utterances[n_train:n_train + n_dev],
        "test": all_utterances[n_train + n_dev:],
    }
    
    logger.info(f"Split into: train={len(split_map['train'])}, dev={len(split_map['dev'])}, test={len(split_map['test'])}")
    
    # Copy files to final split directories (optimized parallel copying)
    for split, utt_list in split_map.items():
        logger.info(f"Copying files for {split} split...")
        
        # Process in parallel batches for much faster copying
        batch_size = 1000  # Process 1000 files per batch
        copy_tasks = []
        
        for i in range(0, len(utt_list), batch_size):
            batch = utt_list[i:i + batch_size]
            copy_tasks.append((batch, output_root, split))
        
        total_copied = 0
        max_copy_workers = min(8, max(1, len(copy_tasks)))  # Ensure at least 1 worker
        
        if len(copy_tasks) == 0:
            logger.warning(f"No files to copy for {split} split")
            continue
        
        with ThreadPoolExecutor(max_workers=max_copy_workers) as executor:
            future_to_batch = {executor.submit(copy_files_batch, task): task for task in copy_tasks}
            
            for future in tqdm(as_completed(future_to_batch), total=len(copy_tasks),
                              desc=f"Copying {split} files (batched)"):
                try:
                    copied_count, errors = future.result()
                    total_copied += copied_count
                    
                    # Log some errors if any
                    if errors and len(errors) < 5:
                        for error in errors[:2]:
                            logger.warning(f"Copy error: {error}")
                            
                except Exception as e:
                    logger.error(f"Failed to process copy batch: {e}")
        
        logger.info(f"Successfully copied {total_copied} files for {split} split")
    
    # Keep temp directory for safety (comment out cleanup)
    # logger.info("Cleaning up temporary files...")
    # if os.path.exists(temp_dir):
    #     shutil.rmtree(temp_dir)  # Much faster than individual file removal
    logger.info(f"Keeping temporary files in: {temp_dir}")
    
    # Report results
    logger.info(f"\nEmoNet dataset download and split complete!")
    logger.info(f"Total utterances processed: {len(all_utterances)}")
    logger.info(f"Data splits saved to: {output_root}")
    
    # Show final structure
    logger.info(f"\nFinal directory structure:")
    logger.info(f"  {output_root}/")
    
    total_files = 0
    for split in splits:
        split_path = os.path.join(output_root, split)
        if os.path.exists(split_path):
            wav_files = [f for f in os.listdir(split_path) if f.endswith('.wav')]
            logger.info(f"    {split}/ ({len(wav_files)} wav files)")
            total_files += len(wav_files)
    
    logger.info(f"\nTotal audio files in splits: {total_files}")
    logger.info("✅ Dataset is ready for training!")
    
    return total_files


if __name__ == "__main__":
    # Configuration
    data_dir = "/tsi/hi-paris/tts/Luka/data"
    
    # Custom splits (optional)
    splits = {"train": 0.9, "dev": 0.05, "test": 0.05}
    
    # Set multiprocessing start method for compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    # Set to True to skip download and use existing temp data
    skip_download = True  # Change to False if you want fresh download
    
    total_files = download_process_and_split_emonet(data_dir, splits, skip_download)
    
    logger.info(f"\nComplete workflow finished! Processed {total_files} audio files.")
    logger.info("Ready for stage 0 training pipeline.")
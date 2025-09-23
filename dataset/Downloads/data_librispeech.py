import os
import argparse
import soundfile as sf
import librosa
from tqdm import tqdm
from datasets import load_dataset, Audio
from huggingface_hub import login
import numpy as np
import urllib.request




login("hf_key")

# --- Configuration pour Librispeech uniquement ---
DATASET_CONFIG = {
    "facebook/multilingual_librispeech": {
        "name": "french", "split": "train", "audio_col": "audio", "text_col": "transcript"
    }
}

TARGET_SAMPLING_RATE = 16000

"""
Tried to determine the gender of the speaker wrt the pitch, but it was not very reliable and dimn't work well on small tests...
I will then switch to something a bit more complex, but more reliable...
"""

def parse_speakers_txt(filepath):
    speaker_gender = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith(';'):
                parts = line.split("|")
                speaker_id = parts[0].strip()
                gender = parts[1].strip().lower()
                speaker_gender[speaker_id] = gender
    return speaker_gender

def collect_and_process(total_hours_target):
    output_dir = os.path.join("dataset", "data", f"librispeech_fr")
    print(f"Will collect at most {total_hours_target}h from librispeech")
    print(f"Files will be saved in: '{output_dir}'")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Outoput folder: {output_dir}")
    total_target_seconds = total_hours_target * 3600
    with tqdm(total=total_target_seconds, unit="s", desc="Total progress", unit_scale=True) as pbar_main:
        speaker_file_counts = {}
        for name, config in DATASET_CONFIG.items():
            print(f"\n---> Treating : {name}")
            dataset = load_dataset(name, config['name'], split=config['split'], streaming=True)
            dataset = dataset.cast_column(config['audio_col'], Audio(sampling_rate=TARGET_SAMPLING_RATE))
            features = dataset.features
            # print(features) # No gender specified
            for sample in dataset:
                if pbar_main.n >= pbar_main.total:
                    break
                try:
                    audio_data = sample[config['audio_col']]['array']
                    text_col = config['text_col']
                    if text_col not in sample:
                        print(f"Ignored key: '{text_col}'. Available ones were: {list(sample.keys())}")
                        continue
                    text_data = sample[text_col]
                    if not text_data or not str(text_data).strip():
                        continue
                    if audio_data.ndim > 1:
                        audio_data = librosa.to_mono(audio_data.T)
                    speaker_id = sample.get('speaker_id', 'unknown')
                    speaker_dir = os.path.join("dataset", "data", f"librispeech_speaker_{speaker_id}")
                    os.makedirs(speaker_dir, exist_ok=True)
                    # Compteur local par speaker
                    if speaker_id not in speaker_file_counts:
                        speaker_file_counts[speaker_id] = 1
                    else:
                        speaker_file_counts[speaker_id] += 1
                    base_filename = f"audio_{speaker_file_counts[speaker_id]:07d}"
                    wav_path = os.path.join(speaker_dir, f"{base_filename}.wav")
                    txt_path = os.path.join(speaker_dir, f"{base_filename}.txt")
                    sf.write(wav_path, audio_data, TARGET_SAMPLING_RATE)
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(str(text_data))
                    duration_seconds = len(audio_data) / TARGET_SAMPLING_RATE
                    pbar_main.update(duration_seconds)
                except Exception as e:
                    raise e
            if pbar_main.n >= pbar_main.total:
                break
    final_hours = pbar_main.n / 3600
    print("\n--- Finished downloading ---")
    print(f"approximatively {final_hours:.2f}h where downloaded")
    print(f"Total number of pairs (.wav et .txt) downloaded: {sum(speaker_file_counts.values())}")
    print(f"The data was saved in: '{output_dir}'")

    print("Now starting to determine the gender of each speakers")
    # Use this file to know the gender of each speakers (for each speaker_id, the gender of the speaker is specified here...)
    url = "https://huggingface.co/datasets/facebook/multilingual_librispeech/raw/main/data/mls_french/metainfo.txt"
    filename = os.path.join("dataset", "data", "librispeech_SPEAKERS.TXT")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")
    genders = parse_speakers_txt(filename)
    for speaker_folder in tqdm(os.listdir("dataset/data")):
        speaker_path = os.path.join("dataset", "data", speaker_folder)
        if not speaker_folder.startswith("librispeech_speaker_"):
            continue
        if not os.path.isdir(speaker_path):
            continue
        if speaker_path.endswith("m") or speaker_path.endswith("f"):
            continue
        speaker_id = speaker_folder.split("speaker_")[1]
        gender = genders[speaker_id]
        new_speaker_path = os.path.join("dataset", "data", f"{speaker_folder}{gender}")
        if os.path.exists(new_speaker_path):
            # Remove the old folder if it exists
            print(f"Removing existing folder: {new_speaker_path}")
            os.rmdir(new_speaker_path)
        # Rename the folder to include
        os.rename(speaker_path, new_speaker_path)
    print("finished determining the gender of each speakers")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hours",
        type=int,
        default=2500,
        help="Maximum number of hours to download. (Default: 2500)"
    )
    args = parser.parse_args()
    collect_and_process(args.hours)

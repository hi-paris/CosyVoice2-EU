import os
from huggingface_hub import hf_hub_download, list_repo_files
import tarfile
import librosa
import soundfile as sf
import json
import shutil


genders = {
    "alloy":"f",
    "ash":"m",
    "ballad":"f",
    "coral":"f",
    "echo":"m",
    "fable":"m",
    "Nova":"f",
    "onyx":"m",
    "sage":"f",
    "shimmer":"f",
    "verse":"m",
}
def download_all_files(data_dir, repo_id, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    print(f"Listing files in repo: {repo_id} ...")
    files = list_repo_files(repo_id, repo_type="dataset")

    print(f"Found {len(files)} files.")
    for file_path in files:
        if file_path[:2] != "fr" :
            continue
        # download only files, skip directories
        print(f"Downloading {file_path} ...")
        speaker = file_path.split("_")[1]
        gender = genders[speaker]
        folder_specific_name = file_path.split("fr_")[1].split(".tar")[0]
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=file_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Saved to {local_path}")


        print("starting to extract...")
        with tarfile.open(local_path, "r") as tar:
            tar.extractall(path=os.path.join(local_dir))
        print(f"Extracted to {os.path.join(local_dir)}")
        os.remove(local_path)
        print(f"Removed the tar file {local_path}")

        print(f"Start processing audios (converting from .mp3 to .wav)...")
        audio_dir = os.path.join(local_dir, file_path[:-4])
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]
        for audio_file in audio_files:
            audio_path = os.path.join(audio_dir, audio_file)
            new_audio_name = f"{audio_file[:-4]}.wav"
            new_audio_path = os.path.join(audio_dir, new_audio_name)
            def mp3_to_wav_librosa(input_mp3, output_wav):
                # sr=None preserves original sample rate
                audio, sr = librosa.load(input_mp3, sr=None)
                sf.write(output_wav, audio, sr)
            mp3_to_wav_librosa(audio_path, new_audio_path)
            os.remove(audio_path)
        print(f"Converted all audio files in {audio_dir} to .wav format.")

        print("Obtaining the text files...")
        json_files = [f for f in os.listdir(audio_dir) if f.endswith('.json')]
        for json_name in json_files:
            json_path = os.path.join(audio_dir, json_name)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            try:
                # Extract the transcription
                text = data.get("annotation")
                if text is None:
                    raise ValueError(f"No 'annotation' field found in {json_path}")
                text = text.split("<transcription_start>")[1].split("</transcription")[0].split("<transcription")[0].strip()
            except:
                print("[WARNING]--", json_name, "is badly formatted, removing the .join file and its associated audio file.")
                os.remove(json_path)
                os.remove(json_path.replace('.json', '.wav'))
                continue
            text = text.replace(".\n", ". ").replace(". \n", ". ")
            if "\n" in text:
                print("[WARNING]--", json_name, "is badly formatted, removing the .join file and its associated audio file.")
                os.remove(json_path)
                os.remove(json_path.replace('.json', '.wav'))
                continue
            # Save to .txt file
            txt_path = json_path.replace('.json', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as out_f:
                out_f.write(text)
            os.remove(os.path.join(audio_dir, json_path))
        print(f"Converted all json files in {audio_dir} to .txt format.")
        
        print(f"Moving the audio-text pairs to the directory of the speaker: {speaker}...")
        speaker_dir = os.path.join(data_dir, f"emonet_{speaker}_{gender}")
        os.makedirs(speaker_dir, exist_ok=True)
        for file_name in os.listdir(audio_dir):
            src_path = os.path.join(audio_dir, file_name)
            dst_path = os.path.join(speaker_dir, f"{folder_specific_name}_{file_name}")
            if os.path.isfile(src_path):
                os.rename(src_path, dst_path)
        print(f"Moved all files from {audio_dir} to {speaker_dir}.")
        os.rmdir(audio_dir)
        print(f"Removed the empty directory {audio_dir}.")

    # Remove the local non-empty directory
    shutil.rmtree(local_dir)
    print(f"Removed the empty directory {local_dir}.")

if __name__ == "__main__":
    dataset_name = "laion/laions_got_talent_enhanced_flash_annotations_and_long_captions"
    # dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join("/tsi/hi-paris/tts/Luka/data")


    repo_id = "laion/laions_got_talent_enhanced_flash_annotations_and_long_captions"
    local_dir =  os.path.join(data_dir, "emonet_full_fr")
    download_all_files(data_dir, repo_id, local_dir)

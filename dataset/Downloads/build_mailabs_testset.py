#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

try:
    import soundfile as sf
except Exception as e:
    print("This script needs the `soundfile` package. Try: pip install soundfile", file=sys.stderr)
    raise

# ---------- Helpers ----------
def human_h(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"

def safe_speaker_id(name: str) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    base = "_".join([p for p in base.split("_") if p])
    return base[:64] if base else "spk"

def short_hash(*parts: str, n: int = 8) -> str:
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:n]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def download_if_needed(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"[download] Using existing: {dest}")
        return dest
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] GET {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        prog = 100 * done / total
                        sys.stdout.write(f"\r[download] {dest.name}: {prog:.1f}%")
                        sys.stdout.flush()
    print()
    return dest

def extract_tgz(tar_path: Path, out_dir: Path) -> Path:
    root_dir = out_dir / tar_path.stem  # e.g., fr_FR
    if root_dir.exists():
        print(f"[extract] Using existing extracted dir: {root_dir}")
        return root_dir
    print(f"[extract] Extracting {tar_path} -> {out_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=out_dir)
    # Try <stem> directly
    candidates = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith(tar_path.stem.split(".")[0])]
    if len(candidates) == 1:
        return candidates[0]
    # Fallback: any dir that contains male/ or female/ (fix precedence)
    for p in out_dir.iterdir():
        if p.is_dir() and ((p / "male").exists() or (p / "female").exists() or (p / "by_book").exists()):
            return p
    return root_dir

def get_wav_duration(wav_path: Path) -> float:
    try:
        info = sf.info(str(wav_path))
        if info.samplerate and info.frames:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    return 0.0

def maybe_resample(in_wav: Path, out_wav: Path, target_sr: int) -> None:
    if target_sr <= 0:
        if in_wav.resolve() != out_wav.resolve():
            shutil.copy2(in_wav, out_wav)
        return
    cmd = ["ffmpeg", "-y", "-i", str(in_wav), "-ar", str(target_sr), "-ac", "1", str(out_wav)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        shutil.copy2(in_wav, out_wav)

# ---------- Robust collector for M-AILABS (male/female + by_book) ----------
def _iter_metadata_jsons(root: Path, gender: str):
    """
    Yield all real 'metadata_mls.json' paths under:
      <root>/<gender>/**/metadata_mls.json
      <root>/by_book/<gender>/**/metadata_mls.json
    Skip AppleDouble files like '._metadata_mls.json'.
    """
    search_roots = []
    direct = root / gender
    by_book = root / "by_book" / gender
    if direct.exists():
        search_roots.append(direct)
    if by_book.exists():
        search_roots.append(by_book)

    for base in search_roots:
        # Normal files
        for p in base.rglob("metadata_mls.json"):
            if p.name.startswith("._"):
                continue
            yield p
        # AppleDouble companions — use their real sibling if present
        for p in base.rglob("._metadata_mls.json"):
            sibling = p.with_name("metadata_mls.json")
            if sibling.exists():
                yield sibling

def _speaker_from_path(meta_json: Path, gender: str) -> str:
    """
    Derive speaker id from path parts: .../<gender>/<speaker>/<book>/metadata_mls.json
    Works for both plain and by_book layouts.
    """
    parts = meta_json.parts
    try:
        idx = len(parts) - 1 - parts[::-1].index(gender)  # index of 'gender'
        speaker = parts[idx + 1] if idx + 1 < len(parts) else "spk"
    except ValueError:
        # gender not found (odd), fallback to parent name
        speaker = meta_json.parent.parent.name
    return f"{gender}_{speaker}"

def collect_mailabs_gender_json(root: Path, gender: str):
    """
    Return entries: list of dicts with keys:
      speaker, wav (Path), base (stem), text (clean/normalized/original), dur (sec)
    """
    entries = []
    for meta_json in _iter_metadata_jsons(root, gender):
        book_dir = meta_json.parent
        wav_root_a = book_dir
        wav_root_b = book_dir / "wavs"
        speaker_tag = _speaker_from_path(meta_json, gender)

        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        for fname, data in meta.items():
            text = (data.get("clean") or data.get("normalized") or data.get("original") or "").strip()
            if not text:
                continue
            wav = (wav_root_b / fname) if (wav_root_b / fname).exists() else (wav_root_a / fname)
            if not wav.exists():
                continue
            dur = get_wav_duration(wav)
            if dur <= 0:
                continue
            entries.append({
                "speaker": speaker_tag,
                "wav": wav,
                "base": Path(fname).stem,
                "text": text,
                "dur": dur,
            })
    return entries

def pick_clips(entries, target_seconds: float, seed: int, min_s: float, max_s: float):
    rng = random.Random(seed)
    candidates = [e for e in entries if min_s <= e["dur"] <= max_s]
    rng.shuffle(candidates)
    picked, total = [], 0.0
    for e in candidates:
        picked.append(e)
        total += e["dur"]
        if total >= target_seconds:
            break
    return picked, total

def write_selection(selection, out_root: Path, lang_tag: str, copy_mode: str, target_sr: int):
    out_base = out_root / f"dataset_test-{lang_tag}" / "test"
    written = 0
    for e in selection:
        spk = safe_speaker_id(e["speaker"])
        utt_hash = short_hash(str(e["wav"]), e["text"])
        leaf = out_base / spk / utt_hash
        ensure_dir(leaf)
        dst_wav = leaf / f"{e['base']}.wav"
        dst_txt = leaf / f"{e['base']}.normalized.txt"

        if copy_mode == "symlink":
            try:
                rel = os.path.relpath(e["wav"], leaf)
                if dst_wav.exists():
                    dst_wav.unlink()
                dst_wav.symlink_to(rel)
            except Exception:
                maybe_resample(e["wav"], dst_wav, target_sr)
        else:
            maybe_resample(e["wav"], dst_wav, target_sr)

        with open(dst_txt, "w", encoding="utf-8") as f:
            f.write(e["text"] + "\n")
        written += 1
    return written

# ---------- Pipeline ----------
def build_for_language_json(
    tgz: Path,
    work_dir: Path,
    out_root: Path,
    lang_tag: str,
    hours: float,
    female_ratio: float,
    seed: int,
    min_s: float,
    max_s: float,
    copy_mode: str,
    target_sr: int,
):
    lang_root = extract_tgz(tgz, work_dir)
    print(f"[{lang_tag}] root: {lang_root}")

    female_entries = collect_mailabs_gender_json(lang_root, "female")
    male_entries   = collect_mailabs_gender_json(lang_root, "male")
    print(f"[{lang_tag}] found entries — female: {len(female_entries)}, male: {len(male_entries)}")

    if not female_entries and not male_entries:
        raise RuntimeError(f"[{lang_tag}] No entries found under male/ or female/ (check layout and JSON files)")

    target_total = hours * 3600.0
    target_f = target_total * max(0.0, min(1.0, female_ratio))
    target_m = max(0.0, target_total - target_f)

    sel_f, tot_f = pick_clips(female_entries, target_f, seed, min_s, max_s) if female_entries else ([], 0.0)
    sel_m, tot_m = pick_clips(male_entries,   target_m, seed + 1, min_s, max_s) if male_entries else ([], 0.0)
    selection = sel_f + sel_m
    random.Random(seed).shuffle(selection)

    spk_count = len({e["speaker"] for e in selection})
    print(f"[{lang_tag}] Picked {len(selection)} clips, {spk_count} speakers, total ~{human_h(tot_f + tot_m)} "
          f"(female ~{human_h(tot_f)}, male ~{human_h(tot_m)})")

    written = write_selection(selection, out_root, lang_tag, copy_mode, target_sr)
    print(f"[{lang_tag}] Wrote {written} items under {out_root}/dataset_test-{lang_tag}/test")
    return (tot_f + tot_m), written, spk_count

def main():
    p = argparse.ArgumentParser(description="Create unseen FR/DE test sets from M-AILABS (~1h/lang, 50% female/male).")
    p.add_argument("--work-dir", type=Path, required=True,
                   help="Cache/work directory for downloads + extraction.")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Parent output directory where dataset_test-FR/DE will be created.")
    p.add_argument("--fr-url", type=str, default="https://ics.tau-ceti.space/data/Training/stt_tts/fr_FR.tgz",
                   help="URL or local path to fr_FR.tgz")
    p.add_argument("--de-url", type=str, default="https://ics.tau-ceti.space/data/Training/stt_tts/de_DE.tgz",
                   help="URL or local path to de_DE.tgz")
    p.add_argument("--hours-per-lang", type=float, default=1.0,
                   help="Target hours per language (default 1.0)")
    p.add_argument("--female-ratio", type=float, default=0.5,
                   help="Fraction of hours-per-lang to take from female/ (rest from male/).")
    p.add_argument("--min-sec", type=float, default=1.0, help="Min clip seconds (default 1.0)")
    p.add_argument("--max-sec", type=float, default=20.0, help="Max clip seconds (default 20.0)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy",
                   help="Copy audio (default) or create symlinks")
    p.add_argument("--target-sr", type=int, default=0,
                   help="Optional resample rate (e.g., 16000/22050/24000). 0 = keep original.")
    args = p.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    def resolve_tgz(src: str) -> Path:
        path = Path(src)
        if path.exists():
            return path
        dest = args.work_dir / Path(src).name
        return download_if_needed(src, dest)

    fr_tgz = resolve_tgz(args.fr_url)
    de_tgz = resolve_tgz(args.de_url)

    # total_fr, n_fr, s_fr = build_for_language_json(
    #     fr_tgz, args.work_dir, args.out_root, "FR",
    #     args.hours_per_lang, args.female_ratio, args.seed, args.min_sec, args.max_sec,
    #     args.copy_mode, args.target_sr
    # )
    total_de, n_de, s_de = build_for_language_json(
        de_tgz, args.work_dir, args.out_root, "DE",
        args.hours_per_lang, args.female_ratio, args.seed + 10, args.min_sec, args.max_sec,
        args.copy_mode, args.target_sr
    )

    print("\n=== Summary ===")
    # print(f"FR: {n_fr} clips, {s_fr} speakers, {human_h(total_fr)}")
    print(f"DE: {n_de} clips, {s_de} speakers, {human_h(total_de)}")
    print(f"Output root: {args.out_root}")

if __name__ == "__main__":
    main()

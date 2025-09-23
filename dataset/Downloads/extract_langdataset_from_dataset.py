#!/usr/bin/env python3
"""
Create single-language TTS datasets by copying only *_DE or *_FR speakers
from bilingual datasets. Uses hardlinks when possible (fast, zero extra
space), and falls back to real copies if cross-device or disallowed.

Features:
- Auto hardlink → copy fallback
- Parallel across datasets and speakers
- Choose language with --lang {DE,FR}
- Safe: no deletions; skips existing targets

Author: you + a helpful assistant
Date: August 2025
"""

import os
import sys
import shutil
import logging
import argparse
from errno import EXDEV, EPERM
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ---------- Copy helpers ----------
def smart_copytree(src: Path, dst: Path):
    """
    Prefer hardlinks (fast, zero-space). If not possible due to cross-device
    (EXDEV) or policy (EPERM), fall back to real copy (copy2).
    """
    try:
        shutil.copytree(src, dst, copy_function=os.link, dirs_exist_ok=False)
    except OSError as e:
        if getattr(e, "errno", None) in (EXDEV, EPERM):
            shutil.copytree(src, dst, copy_function=shutil.copy2, dirs_exist_ok=False)
        else:
            raise

def copy_speaker_dir(speaker_dir: Path, target_split_dir: Path) -> tuple[str, bool, str | None]:
    """Copy one speaker folder; returns (name, ok, error)."""
    target = target_split_dir / speaker_dir.name
    if target.exists():
        return (speaker_dir.name, True, None)  # already there; treat as ok/skip
    try:
        smart_copytree(speaker_dir, target)
        return (speaker_dir.name, True, None)
    except Exception as e:
        return (speaker_dir.name, False, str(e))

# ---------- Core logic ----------
def find_datasets(base_dir: Path) -> list[Path]:
    return sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("tts_dataset_combined_") and d.name.endswith("h")
    )

def process_dataset(
    dataset_dir: Path,
    base_dir: Path,
    lang_suffix: str,
    speaker_workers: int,
):
    """
    Process a single dataset directory:
      - compute new hours (half)
      - create target dataset dir with _{lang} suffix
      - copy only *_{lang} speaker dirs per split, in parallel
    """
    logging.info(f"Processing dataset: {dataset_dir.name}")

    # Parse hours from name
    try:
        hours = int(dataset_dir.name.replace("tts_dataset_combined_", "").replace("h", ""))
        new_hours = hours // 2
    except ValueError:
        logging.error(f"Could not parse hours from: {dataset_dir.name}")
        return

    target_dataset_name = f"tts_dataset_combined_{new_hours}h_{lang_suffix}"
    target_dataset_dir = base_dir / target_dataset_name
    logging.info(f"→ {dataset_dir.name}  →  {target_dataset_name}")

    splits = ("dev", "test", "train")
    for split in splits:
        src_split = dataset_dir / split
        if not src_split.exists():
            logging.warning(f"Missing split: {src_split}")
            continue

        # Collect speakers for chosen language
        speakers = sorted(
            p for p in src_split.iterdir()
            if p.is_dir() and p.name.endswith(f"_{lang_suffix}")
        )
        if not speakers:
            logging.warning(f"No *_{lang_suffix} speakers in {src_split}")
            continue

        dst_split = target_dataset_dir / split
        dst_split.mkdir(parents=True, exist_ok=True)

        logging.info(f"{dataset_dir.name}/{split}: copying {len(speakers)} *_{lang_suffix} speakers")

        results = []
        with ThreadPoolExecutor(max_workers=speaker_workers) as ex:
            futs = [ex.submit(copy_speaker_dir, sp, dst_split) for sp in speakers]
            for fut in as_completed(futs):
                results.append(fut.result())

        ok = sum(1 for _, success, _ in results if success)
        errs = [(n, e) for n, success, e in results if not success]
        logging.info(f"{dataset_dir.name}/{split}: copied {ok}/{len(speakers)} speakers to {dst_split}")

        if errs:
            for n, e in errs[:10]:
                logging.error(f"{dataset_dir.name}/{split}/{n}: {e}")
            if len(errs) > 10:
                logging.error(f"... and {len(errs) - 10} more errors")

def main():
    ap = argparse.ArgumentParser(description="Create single-language TTS datasets via hardlink→copy fallback.")
    ap.add_argument("--base-dir", type=Path, default=Path("/tsi/hi-paris/tts/Luka/data"),
                    help="Base directory containing tts_dataset_combined_*h folders.")
    ap.add_argument("--lang", choices=["DE", "FR"], default="DE",
                    help="Language suffix to select speaker folders and to append to target dataset directory.")
    ap.add_argument("--dataset-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="Max parallel datasets to process.")
    ap.add_argument("--speaker-workers", type=int, default=8,
                    help="Max parallel speaker copies per dataset.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List planned operations without copying.")
    args = ap.parse_args()

    base_dir = args.base_dir
    if not base_dir.exists():
        logging.error(f"Base directory does not exist: {base_dir}")
        sys.exit(1)

    datasets = find_datasets(base_dir)
    if not datasets:
        logging.error("No tts_dataset_combined_*h directories found.")
        sys.exit(1)

    logging.info(f"Found {len(datasets)} dataset(s): {[d.name for d in datasets]}")
    logging.info("Safety: only creates new directories; never deletes or modifies source files.")
    logging.info(f"Mode: auto hardlink→copy fallback; lang={args.lang}")

    if args.dry_run:
        for d in datasets:
            try:
                hours = int(d.name.replace("tts_dataset_combined_", "").replace("h", ""))
                new_hours = hours // 2
            except ValueError:
                logging.warning(f"Skipping (bad hours): {d.name}")
                continue
            target_name = f"tts_dataset_combined_{new_hours}h_{args.lang}"
            logging.info(f"[DRY RUN] {d.name}  →  {target_name}")
        return

    # Parallel across datasets
    dataset_workers = max(1, args.dataset_workers)
    speaker_workers = max(1, args.speaker_workers)

    with ThreadPoolExecutor(max_workers=dataset_workers) as ex:
        futs = [
            ex.submit(process_dataset, d, base_dir, args.lang, speaker_workers)
            for d in datasets
        ]
        for _ in as_completed(futs):
            pass

    logging.info("All done.")

if __name__ == "__main__":
    main()

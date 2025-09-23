#!/usr/bin/env python3
"""
Fish Speech Evaluation Runner

This script provides a unified interface for running Fish Speech evaluation:
- Synthesis only (for fish-speech environment)
- Full evaluation (requires cosyvoice environment for metrics)
- Two-step process (synthesis then metrics)

Usage:
    python run_fishspeech.py --mode synthesis --language fr
    python run_fishspeech.py --mode metrics --language fr  
    python run_fishspeech.py --mode full --language fr
"""

import sys
import os
import argparse
import subprocess
import tempfile
import yaml
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_environment():
    """Check which conda environment we're in"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    return conda_env

def run_synthesis_only(language="fr"):
    """Run Fish Speech synthesis without metrics computation"""
    
    print(f"🐟 Running Fish Speech synthesis for {language.upper()}...")
    print("⚠️  Metrics computation will be skipped")
    
    import logging
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from dataset_reader import DatasetReader
    from baselines_synthesizer import BaselinesSynthesizer
    
    # Read the original config
    config_path = "eval_config_fishspeech.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    testset_root = config.get('dataset', {}).get('testset_root', '/tsi/hi-paris/tts/Luka/data/tts_testset')
    lang_folder = 'dataset_test-FR' if language == 'fr' else ('dataset_test-DE' if language == 'de' else f'dataset_test-{language.upper()}')
    dataset_path = str(Path(testset_root) / lang_folder)
    dataset_reader = DatasetReader(dataset_path)
    
    # Get samples
    dataset_cap = config.get('dataset', {}).get('max_samples_per_split', None)
    baseline_cap = config.get('baselines', {}).get('max_samples', None)
    effective_cap = dataset_cap if dataset_cap not in (None, 'None', '') else baseline_cap
    samples = dataset_reader.get_samples(
        splits=config['dataset']['splits'],
        max_samples_per_split=effective_cap
    )
    
    if not samples:
        print("❌ No samples found!")
        return False
        
    print(f"Loaded {len(samples)} samples")
    
    # Initialize synthesizer
    synthesizer = BaselinesSynthesizer(language=language, device=config['system']['device'])
    
    try:
        # Load Fish Speech model
        fs_cfg = config.get('baselines', {}).get('fishspeech', {})
        synthesizer.load_fishspeech_model(fs_cfg, config['inference']['prompt_audio'])
        
        # Output dir for audio files
        out_dir = Path(config['output']['synth_dir']) / f"baseline_fishspeech_{language}"
        if config['output']['save_audio']:
            out_dir.mkdir(parents=True, exist_ok=True)
        
        # Synthesize
        synth_results = synthesizer.synthesize_batch(
            samples,
            output_dir=str(out_dir) if config['output']['save_audio'] else None,
            model='fishspeech'
        )
        
        # Save synthesis results with timing info (this is the key!)
        synthesis_rows = []
        for sample, syn_result in zip(samples, synth_results):
            row = {
                'utterance_id': sample['utterance_id'],
                'speaker_id': sample['speaker_id'],
                'split': sample['split'],
                'reference_text': sample.get('text', ''),
                'audio_path': syn_result['audio_path'],
                'synthesis_time': syn_result['synthesis_time'],
                'sample_rate': syn_result['sample_rate'],
                'error': syn_result['error'],
                'success': syn_result['audio_tensor'] is not None
            }
            synthesis_rows.append(row)
        
        # Save synthesis results to CSV
        results_dir = Path(config['output']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        synthesis_csv = results_dir / f"baseline_fishspeech_{language}_synthesis.csv"
        
        df_synthesis = pd.DataFrame(synthesis_rows)
        df_synthesis.to_csv(synthesis_csv, index=False)
        
        success_count = len([r for r in synth_results if r['audio_tensor'] is not None])
        print(f"✅ Fish Speech synthesis completed for {language}")
        print(f"📊 Success: {success_count}/{len(synth_results)} samples")
        print(f"💾 Synthesis results saved to: {synthesis_csv}")
        return True
        
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        return False
    finally:
        synthesizer.cleanup()

def run_metrics_only(language="fr", results_dir=None, synth_dir=None):
    """Run metrics computation for existing Fish Speech audio files"""
    
    print(f"📊 Computing Fish Speech metrics for {language.upper()}...")
    
    # Default paths
    if not results_dir:
        results_dir = "/tsi/hi-paris/tts/Luka/evaluation-without-hint/results"
    if not synth_dir:
        synth_dir = "/tsi/hi-paris/tts/Luka/evaluation-without-hint/synthesis_output"
    
    import logging
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from metrics_computer import MetricsComputer
    from dataset_reader import DatasetReader
    
    results_dir = Path(results_dir)
    
    # Check if synthesis results exist
    synthesis_csv = results_dir / f"baseline_fishspeech_{language}_synthesis.csv"
    if not synthesis_csv.exists():
        print(f"❌ Synthesis results not found: {synthesis_csv}")
        print("Please run synthesis first with --mode synthesis")
        return False
    
    # Load synthesis results
    print("Loading synthesis results...")
    df_synthesis = pd.read_csv(synthesis_csv)
    print(f"Loaded {len(df_synthesis)} synthesis records")
    
    # Check audio directory exists
    fishspeech_audio_dir = Path(synth_dir) / f"baseline_fishspeech_{language}"
    if not fishspeech_audio_dir.exists():
        print(f"❌ Fish Speech audio directory not found: {fishspeech_audio_dir}")
        print("Please run synthesis first with --mode synthesis")
        return False
    
    # Load dataset to get ground truth
    print("Loading dataset...")
    testset_root = f"/tsi/hi-paris/tts/Luka/data/tts_testset/dataset_test-{language.upper()}"
    dataset_reader = DatasetReader(testset_root)
    samples = dataset_reader.get_samples(splits=["test"], max_samples_per_split=None)
    
    # Create sample lookup
    sample_map = {s['utterance_id']: s for s in samples}
    print(f"Loaded {len(samples)} dataset samples")
    
    # Initialize metrics computer
    model_dir = "/home/infres/horstmann-24/TTS2/cosy_repo/pretrained_models/CosyVoice2-0.5B"
    metrics_computer = MetricsComputer(model_dir=model_dir, language=language)
    
    # Collect results in the same format as run_baseline_evaluation.py
    metrics_rows = []
    
    print("Computing metrics...")
    for i, synth_row in df_synthesis.iterrows():
        utterance_id = synth_row['utterance_id']
        
        # Get corresponding sample
        if utterance_id not in sample_map:
            print(f"⚠️  Skipping {utterance_id}: not found in dataset")
            continue
            
        sample = sample_map[utterance_id]
        
        # Base row matching run_baseline_evaluation.py format exactly
        row_base = {
            'model': 'baseline_fishspeech',
            'hours': 0,
            'language': language,
            'backbone': 'baseline',
            'utterance_id': utterance_id,
            'speaker_id': synth_row['speaker_id'],
            'split': synth_row['split'],
            'reference_text': synth_row['reference_text'],
            'whisper_transcript': '',
            'ref_norm': '',
            'hyp_norm': '',
            'synthesis_time': synth_row['synthesis_time'],  # From synthesis results!
            'rtf': np.nan  # Will compute below
        }
        
        # Check if synthesis was successful
        if not synth_row['success'] or pd.isna(synth_row['audio_path']):
            print(f"⚠️  Skipping {utterance_id}: synthesis failed")
            # Add NaN metrics for failed synthesis
            for metric in ['mcd', 'wer', 'wer_norm', 'cer', 'cer_norm', 'secs', 'gpe', 'f0_rmse_hz', 'f0_corr', 'vuv']:
                row_base[metric] = np.nan
            metrics_rows.append(row_base)
            continue
        
        # Get audio path
        synth_audio_path = Path(synth_row['audio_path'])
        if not synth_audio_path.exists():
            print(f"⚠️  Skipping {utterance_id}: audio file not found at {synth_audio_path}")
            # Add NaN metrics for missing file
            for metric in ['mcd', 'wer', 'wer_norm', 'cer', 'cer_norm', 'secs', 'gpe', 'f0_rmse_hz', 'f0_corr', 'vuv']:
                row_base[metric] = np.nan
            metrics_rows.append(row_base)
            continue
        
        try:
            # Compute RTF from synthesis results
            try:
                if synth_row['sample_rate'] and synth_row['synthesis_time'] > 0:
                    # Estimate audio duration from file (more reliable)
                    import librosa
                    audio, sr = librosa.load(str(synth_audio_path), sr=None)
                    audio_duration = len(audio) / sr
                    if audio_duration > 0:
                        row_base['rtf'] = float(synth_row['synthesis_time']) / audio_duration
            except Exception:
                pass
            
            # Compute metrics for this sample
            metrics = {}
            
            # MCD
            metrics['mcd'] = metrics_computer.compute_mcd(sample['wav_path'], str(synth_audio_path))
            
            # WER and related metrics  
            wer_pack = metrics_computer.compute_wer_and_norm_with_transcript(sample['text'], str(synth_audio_path))
            metrics['wer'] = wer_pack['wer']
            metrics['wer_norm'] = wer_pack['wer_norm']
            metrics['cer'] = wer_pack['cer']
            metrics['cer_norm'] = wer_pack['cer_norm']
            row_base['whisper_transcript'] = wer_pack['hyp']
            row_base['ref_norm'] = wer_pack['ref_norm']
            row_base['hyp_norm'] = wer_pack['hyp_norm']
            
            # Speaker similarity
            metrics['secs'] = metrics_computer.compute_secs(sample['wav_path'], str(synth_audio_path))
            
            # Pitch metrics
            pitch = metrics_computer.compute_pitch_metrics(sample['wav_path'], str(synth_audio_path))
            metrics['gpe'] = pitch['gpe']
            metrics['f0_rmse_hz'] = pitch['f0_rmse_hz']
            metrics['f0_corr'] = pitch['f0_corr']
            metrics['vuv'] = pitch['vuv']
            
            # Merge base row with computed metrics
            row = {**row_base, **metrics}
            metrics_rows.append(row)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(df_synthesis)} samples")
                
        except Exception as e:
            logging.error(f"Failed to compute metrics for {utterance_id}: {e}")
            # Add NaN metrics for failed computation
            for metric in ['mcd', 'wer', 'wer_norm', 'cer', 'cer_norm', 'secs', 'gpe', 'f0_rmse_hz', 'f0_corr', 'vuv']:
                row_base[metric] = np.nan
            metrics_rows.append(row_base)
    
    if not metrics_rows:
        print("❌ No metrics computed - no valid audio files found")
        return False
    
    # Save results in the same format as run_baseline_evaluation.py
    df = pd.DataFrame(metrics_rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / f"baseline_fishspeech_{language}_metrics.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Metrics computed for {len(metrics_rows)} samples")
    print(f"📁 Results saved to: {output_file}")
    
    # Print summary statistics
    if len(metrics_rows) > 0:
        print("\n📈 Summary Statistics:")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        summary = df[numeric_cols].describe()
        print(summary.round(3))
    
    return True

def run_full_evaluation(language="fr"):
    """Run complete Fish Speech evaluation (synthesis + metrics)"""
    
    env = check_environment()
    print(f"🐟 Running complete Fish Speech evaluation for {language.upper()}")
    print(f"Current environment: {env}")
    
    if env == "fish-speech":
        print("🔄 Two-step process required (different environments)")
        
        # Step 1: Synthesis
        print("\n🎤 Step 1: Running synthesis...")
        if not run_synthesis_only(language):
            return False
        
        # Step 2: Switch environment and run metrics
        print("\n📊 Step 2: Switching to cosyvoice environment for metrics...")
        script_path = Path(__file__).parent / "run_fishspeech.py"
        cmd = [
            "bash", "-c", 
            f"eval \"$(conda shell.bash hook)\" && conda activate cosyvoice && python {script_path} --mode metrics --language {language}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Complete evaluation finished successfully")
            return True
        else:
            print(f"❌ Metrics computation failed: {result.stderr}")
            return False
            
    elif env == "cosyvoice":
        print("⚠️  Running in cosyvoice environment - can only compute metrics")
        print("Run synthesis first in fish-speech environment")
        return run_metrics_only(language)
        
    else:
        print(f"❌ Unknown environment: {env}")
        print("Please activate either 'fish-speech' or 'cosyvoice' environment")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fish Speech evaluation runner")
    parser.add_argument("--mode", 
                       choices=["synthesis", "metrics", "full"], 
                       default="full",
                       help="Mode: synthesis only, metrics only, or full evaluation")
    parser.add_argument("--language", 
                       choices=["fr", "de"], 
                       default="fr", 
                       help="Language to process")
    parser.add_argument("--results-dir", 
                       help="Results directory (for metrics mode)")
    parser.add_argument("--synth-dir", 
                       help="Synthesis output directory (for metrics mode)")
    
    args = parser.parse_args()
    
    env = check_environment()
    print(f"Environment: {env}")
    
    if args.mode == "synthesis":
        success = run_synthesis_only(args.language)
    elif args.mode == "metrics":
        success = run_metrics_only(args.language, args.results_dir, args.synth_dir)
    elif args.mode == "full":
        success = run_full_evaluation(args.language)
    
    if not success:
        sys.exit(1)
    
    print("\n🎉 Fish Speech evaluation completed!")
    
    if args.mode in ["metrics", "full"]:
        results_dir = args.results_dir or "/tsi/hi-paris/tts/Luka/evaluation-without-hint/results"
        metrics_file = Path(results_dir) / f"baseline_fishspeech_{args.language}_metrics.csv"
        if metrics_file.exists():
            print(f"📁 Metrics saved to: {metrics_file}")
            print("\nNext steps:")
            print("1. Generate report: python generate_report.py")
            print("2. Generate web charts: python generate_web_charts.py")

if __name__ == "__main__":
    main()

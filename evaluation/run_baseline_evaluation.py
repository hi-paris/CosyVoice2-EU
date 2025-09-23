#!/usr/bin/env python3
"""
Standalone Baseline TTS Evaluation

Runs one or more baseline models (coqui, openvoice, elevenlabs) as specified in eval_config.yaml.
"""

import yaml
import logging
import pandas as pd
from pathlib import Path
import numpy as np
import time
import argparse

from dataset_reader import DatasetReader
from baselines_synthesizer import BaselinesSynthesizer
from metrics_computer import MetricsComputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

def run_baseline_evaluation(config_path: str = 'eval_config.yaml', language_override: str | None = None, hours_override: str | None = None, use_mixed_model: bool = False):
    """Run standalone baseline evaluation for selected baseline models."""

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Allow CLI overrides
    language = (language_override or config.get('language', 'fr')).lower()
    if use_mixed_model:
        # For baselines this only affects naming in downstream aggregation, we keep 'hours': 0
        config['use_mixed_model'] = True
    baselines_cfg = config.get('baselines', {})
    models = baselines_cfg.get('models', ['coqui'])
    logger.info(f"Starting baseline evaluation for: {models}")
    start_time = time.time()

    # Central testset path (same for all runs)
    testset_root = config.get('dataset', {}).get('testset_root', '/tsi/hi-paris/tts/Luka/data/tts_testset')
    lang_folder = 'dataset_test-FR' if language == 'fr' else ('dataset_test-DE' if language == 'de' else f'dataset_test-{language.upper()}')
    dataset_path = str(Path(testset_root) / lang_folder)
    logger.info(f"Loading dataset from central testset: {dataset_path}")
    dataset_reader = DatasetReader(dataset_path)
    # Determine sample cap: prefer dataset.max_samples_per_split; fallback to baselines.max_samples if explicitly provided and dataset config absent
    dataset_cap = config.get('dataset', {}).get('max_samples_per_split', None)
    baseline_cap = baselines_cfg.get('max_samples', None)
    # If dataset_cap is not None use it; else use baseline_cap. If both None -> no cap.
    effective_cap = dataset_cap if dataset_cap not in (None, 'None', '') else baseline_cap
    samples = dataset_reader.get_samples(
        splits=config['dataset']['splits'],
        max_samples_per_split=effective_cap
    )
    if not samples:
        logger.error("No samples found!")
        return
    logger.info(f"Loaded {len(samples)} samples")

    # Metrics computer (shared)
    metrics_computer = MetricsComputer(
        model_dir=list(config['models'].values())[0]['model_dir'],
        language=language
    )
    enabled_metrics = [k for k, v in config['metrics'].items() if v]
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        logger.info(f"\n=== Running baseline model: {model} ===")
        synthesizer = BaselinesSynthesizer(language=language, device=config['system']['device'])
        try:
            # Load model-specific
            if model == 'coqui':
                synthesizer.load_coqui_model()
                synthesizer.load_coqui_prompt_audio(config['inference']['prompt_audio'])
            elif model == 'openvoice':
                ov_cfg = baselines_cfg.get('openvoice', {})
                synthesizer.load_openvoice_model(ov_cfg, config['inference']['prompt_audio'])
            elif model == 'elevenlabs':
                el_cfg = baselines_cfg.get('elevenlabs', {})
                synthesizer.load_elevenlabs_model(el_cfg, config['inference']['prompt_audio'])
            elif model == 'fishspeech':
                fs_cfg = baselines_cfg.get('fishspeech', {})
                synthesizer.load_fishspeech_model(fs_cfg, config['inference']['prompt_audio'])
            else:
                logger.warning(f"Unknown baseline '{model}', skipping")
                continue

            # Output dir
            out_dir = Path(config['output']['synth_dir']) / f"baseline_{model}_{language}"
            if config['output']['save_audio']:
                out_dir.mkdir(parents=True, exist_ok=True)

            # Synthesize
            synth_results = synthesizer.synthesize_batch(
                samples,
                output_dir=str(out_dir) if config['output']['save_audio'] else None,
                model=model
            )

            # Metrics
            logger.info(f"Computing metrics for {model}...")
            metrics_rows = []
            for sample, syn_result in zip(samples, synth_results):
                row_base = {
                    'model': f"baseline_{model}",
                    'hours': 0,
                    'language': language,
                    'backbone': 'baseline',
                    'utterance_id': sample['utterance_id'],
                    'speaker_id': sample['speaker_id'],
                    'split': sample['split'],
                    'reference_text': sample.get('text', ''),
                    'whisper_transcript': ''
                }
                if syn_result['audio_tensor'] is None:
                    for m in enabled_metrics:
                        row_base[m] = np.nan
                    metrics_rows.append(row_base)
                    continue
                try:
                    metrics = {}
                    syn_path = syn_result['audio_path']
                    if config['metrics'].get('mcd', False):
                        metrics['mcd'] = metrics_computer.compute_mcd(sample['wav_path'], syn_path)
                    if config['metrics'].get('wer', False):
                        wer_pack = metrics_computer.compute_wer_and_norm_with_transcript(sample['text'], syn_path)
                        metrics['wer'] = wer_pack['wer']
                        metrics['wer_norm'] = wer_pack['wer_norm']
                        metrics['cer'] = wer_pack['cer']
                        metrics['cer_norm'] = wer_pack['cer_norm']
                        row_base['whisper_transcript'] = wer_pack['hyp']
                        row_base['reference_text'] = sample['text']
                        row_base['ref_norm'] = wer_pack['ref_norm']
                        row_base['hyp_norm'] = wer_pack['hyp_norm']
                    if config['metrics'].get('secs', False):
                        metrics['secs'] = metrics_computer.compute_secs(sample['wav_path'], syn_path)
                    if config['metrics'].get('gpe', False):
                        pitch = metrics_computer.compute_pitch_metrics(sample['wav_path'], syn_path)
                        metrics['gpe'] = pitch['gpe']
                        metrics['f0_rmse_hz'] = pitch['f0_rmse_hz']
                        metrics['f0_corr'] = pitch['f0_corr']
                        metrics['vuv'] = pitch['vuv']

                    gen_sr = syn_result.get('sample_rate', None)
                    rtf = np.nan
                    try:
                        if gen_sr and syn_result['audio_tensor'] is not None:
                            gen_dur = float(syn_result['audio_tensor'].shape[1]) / float(gen_sr)
                            if gen_dur > 0:
                                rtf = float(syn_result['synthesis_time']) / gen_dur
                    except Exception:
                        pass

                    row = {**row_base
                            , 'synthesis_time': syn_result['synthesis_time']
                            , 'rtf': rtf
                            , **metrics}
                    metrics_rows.append(row)
                except Exception as e:  # pragma: no cover
                    logger.error(f"Metrics failed for {sample['utterance_id']} ({model}): {e}")
                    for m in enabled_metrics:
                        row_base[m] = np.nan
                    metrics_rows.append(row_base)

            df = pd.DataFrame(metrics_rows)
            out_csv = results_dir / f"baseline_{model}_{language}_metrics.csv"
            df.to_csv(out_csv, index=False)
            success = len([r for r in synth_results if r['audio_tensor'] is not None])
            logger.info(f"{model} synthesis success: {success}/{len(synth_results)} | saved metrics -> {out_csv}")
            if success > 0:
                # Quick summary
                for metric in ['mcd', 'wer', 'secs', 'gpe']:
                    if metric in df.columns and df[metric].count() > 0:
                        vals = df[metric].dropna()
                        if len(vals) > 0:
                            logger.info(f"{model} {metric.upper()}: {vals.mean():.3f} ± {vals.std():.3f}")
        finally:
            synthesizer.cleanup()

    total = time.time() - start_time
    logger.info(f"All baselines completed in {total:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Standalone Baseline Evaluation (coqui / openvoice / elevenlabs)")
    parser.add_argument(
        '--config',
        default='eval_config.yaml',
        help='Path to evaluation configuration file'
    )
    parser.add_argument(
        '--language', '--lang',
        dest='language',
        choices=['fr', 'de', 'FR', 'DE'],
        help='Language to evaluate (overrides config).'
    )
    parser.add_argument(
        '--hours',
        help='Comma-separated list of hours (ignored for baselines; accepted for CLI symmetry)'
    )
    parser.add_argument(
        '--use-mixed-model',
        action='store_true',
        help='Tag results as mixed-model run for aggregation consistency.'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Run quick test with 2 samples only'
    )
    
    args = parser.parse_args()
    
    if args.test_only:
        from baselines_synthesizer import test_baselines_synthesizer
        test_baselines_synthesizer()
    else:
        run_baseline_evaluation(
            args.config,
            language_override=args.language,
            hours_override=args.hours,
            use_mixed_model=args.use_mixed_model,
        )


if __name__ == "__main__":
    main()

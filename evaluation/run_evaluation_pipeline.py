#!/usr/bin/env python3
"""
Main Evaluation Pipeline for CosyVoice2

Orchestrates the entire evaluation process:
1. Load dataset directly from file structure
2. Synthesize audio with different model configurations  
3. Compute metrics
4. Generate reports
"""

import yaml
import logging
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List
import time
import argparse

from dataset_reader import DatasetReader
from cosyvoice_synthesizer import CosyVoice2Synthesizer
from baselines_synthesizer import BaselinesSynthesizer
from metrics_computer import MetricsComputer

# Configure logging only if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

logger = logging.getLogger(__name__)

# Reduce verbose logging from third-party libraries
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('librosa').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)

class EvaluationPipeline:
    """Main evaluation pipeline."""
    
    def __init__(self, config_path: str, language: str | None = None, hours: List[int] | None = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Allow CLI overrides for language and hours (fallback to config for backward-compat)
        self.language = (language or self.config.get('language', 'fr')).lower()
        self.hours_to_evaluate = hours or self.config.get('hours_to_evaluate', [])
        if isinstance(self.hours_to_evaluate, str):
            # Support comma-separated string
            self.hours_to_evaluate = [int(x) for x in self.hours_to_evaluate.split(',') if x.strip()]

        self.backbone = self.config['inference']['backbone']

        # Central testset structure: /.../tts_testset/dataset_test-FR|DE with split 'test'
        testset_root = self.config.get('dataset', {}).get(
            'testset_root', '/tsi/hi-paris/tts/Luka/data/tts_testset'
        )
        lang_folder = 'dataset_test-FR' if self.language.lower() == 'fr' else (
            'dataset_test-DE' if self.language.lower() == 'de' else f'dataset_test-{self.language.upper()}'
        )
        dataset_path = str(Path(testset_root) / lang_folder)

        self.dataset_readers = {}
        # Same test set for all hours; map each hour to the same reader for simplicity
        reader = DatasetReader(dataset_path)
        for hour in self.hours_to_evaluate:
            self.dataset_readers[hour] = reader
            
        self.metrics_computer = MetricsComputer(
            model_dir=list(self.config['models'].values())[0]['model_dir'],
            language=self.language
        )
        
        # Create output directories
        Path(self.config['output']['results_dir']).mkdir(parents=True, exist_ok=True)
        if self.config['output']['save_audio']:
            Path(self.config['output']['synth_dir']).mkdir(parents=True, exist_ok=True)
    
    def _get_backbone_suffix(self) -> str:
        """Get backbone suffix for run IDs."""
        backbone_map = {
            'blanken': 'bl',
            'hf:Qwen/Qwen2.5-0.5B': 'q25',
            'hf:Qwen/Qwen3-0.6B': 'q3',
            'hf:utter-project/EuroLLM-1.7B-Instruct': 'eu',
            'hf:mistralai/Mistral-7B-v0.3': 'mi'
        }
        return backbone_map.get(self.backbone, 'bl')  # default to blanken
    
    def _resolve_model_config(self, model_config: Dict, hours: int) -> Dict:
        """Resolve model config with automatic run ID generation."""
        config = model_config.copy()
        config['backbone'] = self.backbone

        backbone_suffix = self._get_backbone_suffix()
        language_code = 'mix' if self.config.get('use_mixed_model', False) else self.language.upper()
        hours_for_id = hours * 2 if language_code == 'mix' else hours

        # Do not override run-ids for the original/pretrained model
        setting = config.get('setting', '')
        if setting != 'original':
            config['llm_run_id'] = f"{hours_for_id}-averaged-{backbone_suffix}-{language_code}"
            config['flow_run_id'] = f"{hours_for_id}-averaged-{backbone_suffix}-{language_code}"

        return config
    
    def load_samples(self, hours: int) -> List[Dict]:
        """Load samples from dataset for specific hour configuration."""
        logger.info(f"Loading dataset samples for {hours}h {self.language.upper()} (central testset)...")
        
        samples = self.dataset_readers[hours].get_samples(
            splits=self.config['dataset']['splits'],
            max_samples_per_split=self.config['dataset'].get('max_samples_per_split')
        )
        
        logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    def _build_inference_config(self) -> Dict:
        # Derive inference config and include evaluation-only optimizations
        inf = dict(self.config['inference'])
        # Optional improvements with safe defaults
        inf.setdefault('workers', self.config.get('system', {}).get('num_workers', 1))
        inf.setdefault('timeout_s', 45)
        inf.setdefault('warmup', True)
        # Prompt caching optional fields (no-op if not provided)
        inf.setdefault('prompt_text', self.config.get('inference', {}).get('prompt_text', ''))
        inf.setdefault('zero_shot_spk_id', self.config.get('inference', {}).get('zero_shot_spk_id', ''))
        # vLLM optional (LLM dynamic batching under concurrency)
        # This flag is read from model config when loading the model

        # store language and language hint boolean in inference config
        inf['language'] = self.language
        inf['add_language_hint'] = self.config.get('add_language_hint', False)

        return inf
    
    def _get_prompt_audio_for_language(self) -> str:
        """Get appropriate prompt audio based on language or use default."""
        prompt_config = self.config['inference']['prompt_audio']
        
        # If prompt_audio is a dict with language keys, use it
        if isinstance(prompt_config, dict):
            if self.language and self.language in prompt_config:
                return prompt_config[self.language]
            # Fallback to default or first available
            return prompt_config.get('default', next(iter(prompt_config.values())))
        
        # If it's a string, use it directly
        return prompt_config

    def synthesize_for_model(self, model_name: str, model_config: Dict, 
                           samples: List[Dict], hours: int) -> List[Dict]:
        """Synthesize audio for a specific model configuration."""
        logger.info(f"Synthesizing with model: {model_name} ({hours}h)")
        
        # Initialize synthesizer
        synthesizer = CosyVoice2Synthesizer(
            model_config=model_config,
            device=self.config['system']['device']
        )
        
        try:
            # Get appropriate prompt audio
            prompt_audio = self._get_prompt_audio_for_language()
            logger.info(f"Using prompt audio for language '{self.language}': {prompt_audio}")
            
            # Load prompt audio
            synthesizer.load_prompt_audio(prompt_audio)
            
            # Build inference options
            inference_cfg = self._build_inference_config()
            
            # Prepare output directory
            synth_output_dir = None
            if self.config['output']['save_audio']:
                synth_output_dir = Path(self.config['output']['synth_dir']) / f"{model_name}_{hours}h_{self.language}_model-{self.language if not self.config.get('use_mixed_model', False) else 'mix'}"
                synth_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Synthesize batch
            results = synthesizer.synthesize_batch(
                samples=samples,
                inference_config=inference_cfg,
                output_dir=str(synth_output_dir) if synth_output_dir else None
            )
            
            return results
            
        finally:
            # Clean up model to avoid threading issues
            try:
                synthesizer.cleanup()
            except Exception as e:
                logger.warning(f"Error during model cleanup: {e}")
            del synthesizer
    
    def compute_metrics_for_model(self, model_name: str, samples: List[Dict], 
                                synthesis_results: List[Dict], hours: int) -> pd.DataFrame:
        """Compute metrics for synthesized samples."""
        logger.info(f"Computing metrics for model: {model_name} ({hours}h)")
        
        metrics_data = []
        enabled_metrics = [k for k, v in self.config['metrics'].items() if v]
        
        for sample, syn_result in zip(samples, synthesis_results):
            if syn_result['audio_tensor'] is None:
                # Synthesis failed - record NaN metrics
                metrics_row = {
                    'model': model_name,
                    'hours': hours,
                    'language': self.language,
                    'backbone': self.backbone,
                    'utterance_id': sample['utterance_id'],
                    'speaker_id': sample['speaker_id'],
                    'split': sample['split'],
                    'reference_text': sample.get('text', ''),
                    'whisper_transcript': ''
                }
                for metric in enabled_metrics:
                    metrics_row[metric] = np.nan
                metrics_data.append(metrics_row)
                continue
            
            try:
                # Load reference audio
                import librosa
                ref_audio, ref_sr = librosa.load(sample['wav_path'], sr=16000)
                
                # Convert synthesis result to numpy
                syn_audio = syn_result['audio_tensor'].numpy()
                syn_path  = syn_result['audio_path']
                
                # Compute metrics
                metrics = {}
                if self.config['metrics'].get('mcd', False):
                    metrics['mcd']  = self.metrics_computer.compute_mcd(
                        sample['wav_path'], syn_path)

                if self.config['metrics'].get('wer', False):
                    # Retrieve WER and transcript
                    wer_pack = self.metrics_computer.compute_wer_and_norm_with_transcript(sample['text'], syn_path)
                    metrics['wer'] = wer_pack['wer']
                    metrics['wer_norm'] = wer_pack['wer_norm']
                    metrics['cer'] = wer_pack['cer']
                    metrics['cer_norm'] = wer_pack['cer_norm']
                    metrics['reference_text'] = sample['text']
                    metrics['whisper_transcript'] = wer_pack['hyp']
                    metrics['ref_norm'] = wer_pack['ref_norm']
                    metrics['hyp_norm'] = wer_pack['hyp_norm']
                
                if self.config['metrics'].get('secs', False):
                    metrics['secs'] = self.metrics_computer.compute_secs(
                        sample['wav_path'], syn_path)

                if self.config['metrics'].get('gpe', False):
                    pitch = self.metrics_computer.compute_pitch_metrics(sample['wav_path'], syn_path)
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

                # Create metrics row
                metrics_row = {
                    'model': model_name,
                    'hours': hours,
                    'language': self.language,
                    'backbone': self.backbone,
                    'utterance_id': sample['utterance_id'],
                    'speaker_id': sample['speaker_id'],
                    'split': sample['split'],
                    'synthesis_time': syn_result['synthesis_time'],
                    'rtf': rtf,
                    **metrics
                }
                metrics_data.append(metrics_row)
                
            except Exception as e:
                logger.error(f"Metrics computation failed for {sample['utterance_id']}: {e}")
                # Record failed metrics
                metrics_row = {
                    'model': model_name,
                    'hours': hours,
                    'language': self.language,
                    'backbone': self.backbone,
                    'utterance_id': sample['utterance_id'],
                    'speaker_id': sample['speaker_id'],
                    'split': sample['split'],
                    'reference_text': sample.get('text', ''),
                    'whisper_transcript': ''
                }
                for metric in enabled_metrics:
                    metrics_row[metric] = np.nan
                metrics_data.append(metrics_row)
        
        return pd.DataFrame(metrics_data)
    
    def synthesize_baselines(self, samples: List[Dict]) -> List[Dict]:
        """Synthesize with configured baseline model(s). Returns list of dicts with model key added."""
        if not self.config.get('run_baselines', False):
            logger.info("Baseline evaluation disabled in config")
            return []
        baselines_conf = self.config.get('baselines', {})
        models = baselines_conf.get('models', ['coqui'])
        max_samples = baselines_conf.get('max_samples', len(samples))
        baseline_samples = samples[:max_samples] if max_samples < len(samples) else samples
        all_results = []
        for model in models:
            logger.info(f"Running baseline '{model}' on {len(baseline_samples)} samples")
            synthesizer = BaselinesSynthesizer(language=self.language, device=self.config['system']['device'])
            try:
                if model == 'coqui':
                    synthesizer.load_coqui_model()
                    synthesizer.load_coqui_prompt_audio(self.config['inference']['prompt_audio'])
                elif model == 'openvoice':
                    ov_cfg = baselines_conf.get('openvoice', {})
                    synthesizer.load_openvoice_model(ov_cfg, self.config['inference']['prompt_audio'])
                else:
                    logger.warning(f"Unknown baseline model '{model}' - skipping")
                    continue
                out_dir = None
                if self.config['output']['save_audio']:
                    out_dir = Path(self.config['output']['synth_dir']) / f"baseline_{model}_{self.language}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                results = synthesizer.synthesize_batch(baseline_samples, output_dir=str(out_dir) if out_dir else None, model=model)
                # tag model
                for r in results:
                    r['baseline_model'] = model
                all_results.extend(results)
            finally:
                try:
                    synthesizer.cleanup()
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Error during baseline '{model}' cleanup: {e}")
                del synthesizer
        return all_results
    
    def compute_metrics_for_baselines(self, baseline_results: List[Dict], 
                                    samples: List[Dict]) -> pd.DataFrame:
        """Compute metrics for baseline results (supports multiple baseline models)."""
        if not baseline_results:
            return pd.DataFrame()
        logger.info("Computing metrics for baseline models")
        enabled_metrics = [k for k, v in self.config['metrics'].items() if v]
        metrics_rows = []
        # Map from utterance_id to sample for quick lookup
        sample_map = {s['utterance_id']: s for s in samples}
        for res in baseline_results:
            sample = sample_map.get(res['utterance_id'])
            if not sample:
                continue
            model_tag = f"baseline_{res.get('baseline_model','unknown')}"
            if res['audio_tensor'] is None:
                row = {
                    'model': model_tag,
                    'hours': 0,
                    'language': self.language,
                    'backbone': 'baseline',
                    'utterance_id': sample['utterance_id'],
                    'speaker_id': sample['speaker_id'],
                    'split': sample['split'],
                    'reference_text': sample.get('text',''),
                    'whisper_transcript': ''
                }
                for m in enabled_metrics:
                    row[m] = np.nan
                metrics_rows.append(row)
                continue
            try:
                syn_path = res['audio_path']
                metrics = {}
                if self.config['metrics'].get('mcd', False):
                    metrics['mcd'] = self.metrics_computer.compute_mcd(sample['wav_path'], syn_path)
                if self.config['metrics'].get('wer', False):
                    wer_pack = self.metrics_computer.compute_wer_and_norm_with_transcript(sample['text'], syn_path)
                    metrics['wer'] = wer_pack['wer']
                    metrics['wer_norm'] = wer_pack['wer_norm']
                    metrics['cer'] = wer_pack['cer']
                    metrics['cer_norm'] = wer_pack['cer_norm']
                    metrics['reference_text'] = sample['text']
                    metrics['whisper_transcript'] = wer_pack['hyp']
                    metrics['ref_norm'] = wer_pack['ref_norm']
                    metrics['hyp_norm'] = wer_pack['hyp_norm']
                if self.config['metrics'].get('secs', False):
                    metrics['secs'] = self.metrics_computer.compute_secs(sample['wav_path'], syn_path)
                if self.config['metrics'].get('gpe', False):
                    pitch = self.metrics_computer.compute_pitch_metrics(sample['wav_path'], syn_path)
                    metrics['gpe'] = pitch['gpe']
                    metrics['f0_rmse_hz'] = pitch['f0_rmse_hz']
                    metrics['f0_corr'] = pitch['f0_corr']
                    metrics['vuv'] = pitch['vuv']
                row = {
                    'model': model_tag,
                    'hours': 0,
                    'language': self.language,
                    'backbone': 'baseline',
                    'utterance_id': sample['utterance_id'],
                    'speaker_id': sample['speaker_id'],
                    'split': sample['split'],
                    'synthesis_time': res['synthesis_time'],
                    **metrics
                }
                metrics_rows.append(row)
            except Exception as e:  # pragma: no cover
                logger.error(f"Baseline metrics computation failed for {sample['utterance_id']}: {e}")
                row = {
                    'model': model_tag,
                    'hours': 0,
                    'language': self.language,
                    'backbone': 'baseline',
                    'utterance_id': sample['utterance_id'],
                    'speaker_id': sample['speaker_id'],
                    'split': sample['split'],
                    'reference_text': sample.get('text',''),
                    'whisper_transcript': ''
                }
                for m in enabled_metrics:
                    row[m] = np.nan
                metrics_rows.append(row)
        df = pd.DataFrame(metrics_rows)
        if not df.empty:
            # Save per model
            out_dir = Path(self.config['output']['results_dir'])
            out_dir.mkdir(parents=True, exist_ok=True)
            for model_name, group in df.groupby('model'):
                path = out_dir / f"{model_name}_{self.language}_metrics.csv"
                group.to_csv(path, index=False)
                logger.info(f"Saved baseline results to: {path}")
        return df

    def generate_summary_report(self, all_metrics_df: pd.DataFrame):
        """Generate comprehensive summary reports."""
        logger.info("Generating summary reports...")
        
        # 1. Component Analysis Report (across different model configurations)
        component_summary = self._generate_component_analysis(all_metrics_df)
        
        # 2. Training Duration Analysis (full_finetuned across different hours)
        duration_summary = self._generate_duration_analysis(all_metrics_df)
        
        # Save summaries
        results_dir = Path(self.config['output']['results_dir'])
        
        component_path = results_dir / 'component_analysis.csv'
        component_summary.to_csv(component_path, index=False)
        
        duration_path = results_dir / 'duration_analysis.csv'  
        duration_summary.to_csv(duration_path, index=False)
        
        # Generate markdown reports
        self._generate_markdown_reports(component_summary, duration_summary, results_dir)
        
        logger.info(f"Component analysis saved to: {component_path}")
        logger.info(f"Duration analysis saved to: {duration_path}")
        
        return component_summary, duration_summary
    
    def _generate_component_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze impact of different model components."""
        component_stats = []
        
        # Group by model and language, compute stats across all hours
        for model in df['model'].unique():
            for lang in df['language'].unique():
                model_data = df[(df['model'] == model) & (df['language'] == lang)]
                if len(model_data) == 0:
                    continue
                    
                stats_row = {
                    'model': model,
                    'language': lang,
                    'total_samples': len(model_data),
                    'success_rate': len(model_data.dropna(subset=['mcd'])) / len(model_data),
                }
                
                # Compute aggregated metrics across all hours
                for metric in ['mcd', 'wer', 'secs', 'gpe']:
                    if metric in model_data.columns:
                        values = model_data[metric].dropna()
                        if len(values) > 0:
                            stats_row[f'{metric}_mean'] = values.mean()
                            stats_row[f'{metric}_std'] = values.std()
                        else:
                            stats_row[f'{metric}_mean'] = np.nan
                            stats_row[f'{metric}_std'] = np.nan
                
                component_stats.append(stats_row)
        
        return pd.DataFrame(component_stats)
    
    def _generate_duration_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze how metrics evolve with training duration for full_finetuned model."""
        duration_stats = []
        
        # Focus on full_finetuned model only
        full_model_data = df[df['model'] == 'full_finetuned']
        
        for lang in full_model_data['language'].unique():
            for hours in sorted(full_model_data['hours'].unique()):
                hour_data = full_model_data[
                    (full_model_data['language'] == lang) & 
                    (full_model_data['hours'] == hours)
                ]
                if len(hour_data) == 0:
                    continue
                    
                stats_row = {
                    'language': lang,
                    'hours': hours,
                    'total_samples': len(hour_data),
                    'success_rate': len(hour_data.dropna(subset=['mcd'])) / len(hour_data),
                }
                
                # Compute metrics
                for metric in ['mcd', 'wer', 'secs', 'gpe']:
                    if metric in hour_data.columns:
                        values = hour_data[metric].dropna()
                        if len(values) > 0:
                            stats_row[f'{metric}_mean'] = values.mean()
                            stats_row[f'{metric}_std'] = values.std()
                        else:
                            stats_row[f'{metric}_mean'] = np.nan
                            stats_row[f'{metric}_std'] = np.nan
                
                duration_stats.append(stats_row)
        
        return pd.DataFrame(duration_stats)
    
    def _generate_markdown_reports(self, component_df: pd.DataFrame, duration_df: pd.DataFrame, output_dir: Path):
        """Generate markdown reports."""
        report_path = output_dir / 'evaluation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# CosyVoice2 Evaluation Report\n\n")
            f.write(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Language:** {self.language.upper()}\n")
            f.write(f"**Backbone:** {self.backbone}\n")
            f.write(f"**Training Hours Evaluated:** {', '.join(map(str, self.hours_to_evaluate))}\n")
            if self.config.get('run_baselines', False):
                baseline_models = self.config.get('baselines', {}).get('models', [])
                f.write(f"**Baseline Models:** {', '.join(baseline_models)}\n")
            f.write("\n")
            
            # Baseline Analysis Section (if baselines were run)
            baseline_data = component_df[component_df['model'].str.startswith('baseline_')]
            if not baseline_data.empty:
                f.write("## 0. Baseline Models Performance\n\n")
                f.write("Performance of state-of-the-art baseline TTS models for comparison.\n\n")
                
                f.write("| Model | Language | Success Rate | MCD ↓ | WER ↓ | SECS ↑ | GPE ↓ |\n")
                f.write("|-------|----------|--------------|--------|--------|---------|--------|\n")
                
                for _, row in baseline_data.iterrows():
                    model = row['model'].replace('baseline_', '').title()  # Clean model name
                    lang = row['language']
                    success_rate = f"{row['success_rate']:.1%}"
                    mcd = f"{row.get('mcd_mean', np.nan):.2f} ± {row.get('mcd_std', np.nan):.2f}" if not pd.isna(row.get('mcd_mean')) else "N/A"
                    wer = f"{row.get('wer_mean', np.nan):.1f} ± {row.get('wer_std', np.nan):.1f}" if not pd.isna(row.get('wer_mean')) else "N/A"
                    secs = f"{row.get('secs_mean', np.nan):.3f} ± {row.get('secs_std', np.nan):.3f}" if not pd.isna(row.get('secs_mean')) else "N/A"
                    gpe = f"{row.get('gpe_mean', np.nan):.1f} ± {row.get('gpe_std', np.nan):.1f}" if not pd.isna(row.get('gpe_mean')) else "N/A"
                    
                    f.write(f"| {model} | {lang} | {success_rate} | {mcd} | {wer} | {secs} | {gpe} |\n")
                f.write("\n")
            
            # Component Analysis Section (excluding baselines)
            cosyvoice_data = component_df[~component_df['model'].str.startswith('baseline_')]
            f.write("## 1. Component Impact Analysis\n\n")
            f.write("This analysis shows the impact of different model components on synthesis quality.\n\n")
            
            f.write("### Component Performance Summary\n\n")
            f.write("| Model | Language | Success Rate | MCD ↓ | WER ↓ | SECS ↑ | GPE ↓ |\n")
            f.write("|-------|----------|--------------|--------|--------|---------|--------|\n")
            
            for _, row in cosyvoice_data.iterrows():
                model = row['model']
                lang = row['language']
                success_rate = f"{row['success_rate']:.1%}"
                mcd = f"{row.get('mcd_mean', np.nan):.2f} ± {row.get('mcd_std', np.nan):.2f}" if not pd.isna(row.get('mcd_mean')) else "N/A"
                wer = f"{row.get('wer_mean', np.nan):.1f} ± {row.get('wer_std', np.nan):.1f}" if not pd.isna(row.get('wer_mean')) else "N/A"
                secs = f"{row.get('secs_mean', np.nan):.3f} ± {row.get('secs_std', np.nan):.3f}" if not pd.isna(row.get('secs_mean')) else "N/A"
                gpe = f"{row.get('gpe_mean', np.nan):.1f} ± {row.get('gpe_std', np.nan):.1f}" if not pd.isna(row.get('gpe_mean')) else "N/A"
                
                f.write(f"| {model} | {lang} | {success_rate} | {mcd} | {wer} | {secs} | {gpe} |\n")
            
            # Duration Analysis Section
            f.write("\n## 2. Training Duration Analysis\n\n")
            f.write("This analysis shows how synthesis quality evolves with training duration for the full fine-tuned model.\n\n")
            
            f.write("### Quality vs Training Duration\n\n")
            f.write("| Language | Hours | Success Rate | MCD ↓ | WER ↓ | SECS ↑ | GPE ↓ |\n")
            f.write("|----------|-------|--------------|--------|--------|---------|--------|\n")
            
            for _, row in duration_df.iterrows():
                lang = row['language']
                hours = int(row['hours'])
                success_rate = f"{row['success_rate']:.1%}"
                mcd = f"{row.get('mcd_mean', np.nan):.2f} ± {row.get('mcd_std', np.nan):.2f}" if not pd.isna(row.get('mcd_mean')) else "N/A"
                wer = f"{row.get('wer_mean', np.nan):.1f} ± {row.get('wer_std', np.nan):.1f}" if not pd.isna(row.get('wer_mean')) else "N/A"  
                secs = f"{row.get('secs_mean', np.nan):.3f} ± {row.get('secs_std', np.nan):.3f}" if not pd.isna(row.get('secs_mean')) else "N/A"
                gpe = f"{row.get('gpe_mean', np.nan):.1f} ± {row.get('gpe_std', np.nan):.1f}" if not pd.isna(row.get('gpe_mean')) else "N/A"
                
                f.write(f"| {lang} | {hours}h | {success_rate} | {mcd} | {wer} | {secs} | {gpe} |\n")
            
            f.write("\n## Metrics Legend\n\n")
            f.write("- **MCD**: Mel-Cepstral Distortion (lower is better) - measures spectral distortion\n")
            f.write("- **WER**: Word Error Rate % (lower is better) - transcription accuracy\n") 
            f.write("- **SECS**: Speaker Embedding Cosine Similarity (higher is better) - speaker similarity\n")
            f.write("- **GPE**: Gross Pitch Error % (lower is better) - pitch accuracy\n")
            f.write("- Values shown as mean ± standard deviation\n")
        
        logger.info(f"Markdown report saved to: {report_path}")

    def run_evaluation(self):
        """Run the complete evaluation pipeline across multiple hours."""
        logger.info("Starting CosyVoice2 evaluation pipeline...")
        start_time = time.time()
        
        # Store all metrics across all hours
        all_metrics = []
        
        # Run baseline evaluation first (using test set from first hour configuration)
        if self.config.get('run_baselines', False):
            logger.info("=== Running Baseline Models Evaluation ===")
            
            # Use first hour configuration for dataset (test set is the same across all hours)
            first_hour = self.hours_to_evaluate[0]
            baseline_samples = self.load_samples(first_hour)
            
            if baseline_samples:
                try:
                    # Run baseline synthesis
                    baseline_results = self.synthesize_baselines(baseline_samples)
                    
                    if baseline_results:
                        # Compute baseline metrics
                        baseline_metrics_df = self.compute_metrics_for_baselines(
                            baseline_results, baseline_samples)
                        
                        if not baseline_metrics_df.empty:
                            all_metrics.append(baseline_metrics_df)
                            logger.info(f"CoquiTTS baseline evaluation completed with {len(baseline_metrics_df)} results")
                
                except Exception as e:
                    logger.error(f"Baseline evaluation failed: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.warning("No samples found for baseline evaluation")
        
        # Evaluate each hour configuration
        for hours in self.hours_to_evaluate:
            logger.info(f"=== Evaluating {hours}h training data ===")
            
            # Load samples for this hour configuration
            samples = self.load_samples(hours)
            
            if not samples:
                logger.warning(f"No samples found for {hours}h! Skipping...")
                continue
            
            # Evaluate each model for this hour configuration
            for model_name, model_config in self.config['models'].items():
                try:
                    # Resolve model config with automatic run IDs
                    resolved_config = self._resolve_model_config(model_config, hours)
                    
                    logger.info(f"Model {model_name} config for {hours}h: "
                              f"llm_run_id={resolved_config.get('llm_run_id')}, "
                              f"flow_run_id={resolved_config.get('flow_run_id')}")
                    
                    # Add small delay between models to avoid threading issues
                    if len(all_metrics) > 0:
                        time.sleep(2)
                    
                    # Synthesize
                    synthesis_results = self.synthesize_for_model(
                        model_name, resolved_config, samples, hours)
                    
                    # Compute metrics
                    metrics_df = self.compute_metrics_for_model(
                        model_name, samples, synthesis_results, hours)
                    all_metrics.append(metrics_df)
                    
                    # Save per-model-hour results
                    model_results_path = Path(self.config['output']['results_dir']) / f"{model_name}_{hours}h_{self.language}_metrics_model-{'mix' if self.config.get('use_mixed_model', False) else self.language}.csv"
                    metrics_df.to_csv(model_results_path, index=False)
                    logger.info(f"Saved {model_name} {hours}h results to: {model_results_path}")
                    
                except Exception as e:
                    logger.error(f"Evaluation failed for model {model_name} {hours}h: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
        
        if not all_metrics:
            logger.error("No successful evaluations!")
            return
        
        # Combine all metrics
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        
        # Save combined results
        combined_path = Path(self.config['output']['results_dir']) / f'all_metrics_{self.language}_{self.backbone.replace(":", "_").replace("/", "_")}.csv'
        combined_metrics.to_csv(combined_path, index=False)
        logger.info(f"Combined metrics saved to: {combined_path}")
        
        # Generate summary reports
        if self.config['output']['generate_report']:
            component_summary, duration_summary = self.generate_summary_report(combined_metrics)
            
            # Print summaries to console
            print("\n" + "="*80)
            print("COMPONENT ANALYSIS SUMMARY")
            print("="*80)
            print(component_summary.to_string(index=False))
            
            print("\n" + "="*80)
            print("DURATION ANALYSIS SUMMARY")
            print("="*80)
            print(duration_summary.to_string(index=False))
        
        total_time = time.time() - start_time
        logger.info(f"Evaluation completed in {total_time:.1f} seconds")


def main():
    parser = argparse.ArgumentParser(description="CosyVoice2 Evaluation Pipeline")
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
        '--use-mixed-model',
        action='store_true',
        help='Use mixed models (trained on FR+DE). Affects run-id suffix to use "mix".'
    )
    parser.add_argument(
        '--hours',
        help='Comma-separated list of training hours to evaluate (overrides config), e.g. 50,100,250'
    )
    parser.add_argument(
        '--add-language-hint',
        action='store_true',
        help='Add language hint to prompt'
    )
    parser.add_argument(
        '--test-dataset', 
        action='store_true',
        help='Test dataset reader only'
    )
    parser.add_argument(
        '--test-synthesis',
        action='store_true', 
        help='Test synthesis only'
    )
    parser.add_argument(
        '--test-metrics',
        action='store_true',
        help='Test metrics only'
    )
    parser.add_argument(
        '--test-baselines',
        action='store_true',
        help='Test baseline models only'
    )
    
    args = parser.parse_args()
    
    if args.test_dataset:
        # Test dataset reader
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Use central testset for the provided or configured language
        lang = (args.language or config.get('language', 'fr')).lower()
        testset_root = config.get('dataset', {}).get('testset_root', '/tsi/hi-paris/tts/Luka/data/tts_testset')
        lang_folder = 'dataset_test-FR' if lang == 'fr' else ('dataset_test-DE' if lang == 'de' else f'dataset_test-{lang.upper()}')
        reader = DatasetReader(str(Path(testset_root) / lang_folder))
        samples = reader.get_samples(['test'], max_samples_per_split=5)
        print(f"Found {len(samples)} samples")
        for sample in samples[:3]:
            print(f"  {sample['utterance_id']}: {sample['text'][:50]}...")
        return
    
    if args.test_synthesis:
        # Test synthesis
        from cosyvoice_synthesizer import test_synthesizer
        test_synthesizer()
        return
    
    if args.test_metrics:
        # Test metrics
        from metrics_computer import test_metrics
        test_metrics()
        return
    
    if args.test_baselines:
        # Test baseline models
        from baselines_synthesizer import test_baselines_synthesizer
        test_baselines_synthesizer()
        return
    
    # Run full evaluation
    # Parse hours string to list[int] if provided
    hours_list = None
    if args.hours:
        hours_list = [int(x) for x in args.hours.split(',') if x.strip()]

    # Build pipeline directly to preserve model order from YAML
    pipeline = EvaluationPipeline(args.config, language=args.language, hours=hours_list)
    if args.use_mixed_model:
        print("Using mixed model.")
        pipeline.config['use_mixed_model'] = True

    if args.add_language_hint:
        print("Adding language hint to prompts.")
        pipeline.config['add_language_hint'] = True

    pipeline.run_evaluation()


if __name__ == "__main__":
    main()

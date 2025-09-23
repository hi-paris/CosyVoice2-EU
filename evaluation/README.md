# CosyVoice2 Evaluation Pipeline

A streamlined evaluation system for CosyVoice2 models that works directly with your file structure.

## Features

## Key Features

- **Direct Dataset Integration**: Works with your existing file structure (no CSV conversion)
- **Multiple Model Configurations**: Evaluates 8 different CosyVoice2 variants
- **Baseline Comparison**: Includes Orpheus TTS and CoquiTTS for SOTA baseline comparison
- **Comprehensive Metrics**: MCD, WER (via Microsoft Phi-4), SECS, GPE/FFE
- **YAML Configuration**: Easy to modify for different datasets and models
- **Robust Error Handling**: Graceful degradation when optional dependencies are missing
- **Minimal Dependencies**: Clean, streamlined codebase

## Quick Start

1. **Install baseline dependencies** (optional):
   ```bash
   pip install -r requirements_baselines.txt
   ```

2. **Test the setup**:
   ```bash
   cd /home/infres/horstmann-24/TTS2/evaluation
   python test_pipeline.py
   ```

3. **Update configuration** (if needed):
   ```bash
   nano eval_config.yaml  # Update paths and model configs
   ```

4. **Test with a small dataset**:
   ```bash
   python run_evaluation_pipeline.py --test-dataset
   ```

5. **Test baseline models** (optional):
   ```bash
   python run_evaluation_pipeline.py --test-baselines
   ```

6. **Run full evaluation**:
   ```bash
   python run_evaluation_pipeline.py
   ```

## Configuration

Edit `eval_config.yaml` to:
- Set dataset paths
- Configure model variants to test
- Enable/disable baseline models (`run_baselines: true/false`)
- Choose which metrics to compute
- Set output directories

## Baseline Models

When `run_baselines: true` is set in the config, the pipeline will also evaluate:

- **CoquiTTS (XTTS v2)**: Popular open-source multilingual voice cloning model

The baseline model is evaluated once on the test set (using the same data as the first hour configuration). This provides a SOTA comparison point for your CosyVoice2 models.

## Model Configurations

The pipeline supports testing different model configurations:
- `pretrained`: Original model
- `llm_only`: Only LLM fine-tuned
- `flow_only`: Only Flow fine-tuned  
- `hifigan_only`: Only HiFiGAN fine-tuned
- `full_finetuned`: All components fine-tuned

## Output

Results are saved to:
- `results/all_metrics.csv`: Per-utterance metrics for all models
- `results/{model}_metrics.csv`: Per-utterance metrics for each model
- `results/summary_report.csv`: Aggregated statistics
- `results/summary_report.md`: Human-readable report
- `synthesis_output/{model}/`: Synthesized audio files (if enabled)

## Files

- `eval_config.yaml`: Main configuration file
- `run_evaluation_pipeline.py`: Main evaluation script
- `dataset_reader.py`: Direct file structure reader
- `cosyvoice_synthesizer.py`: Model wrapper with batch processing
- `metrics_computer.py`: Acoustic and intelligibility metrics
- `test_pipeline.py`: Setup validation script

## Requirements

**Critical dependencies:**
- numpy, pandas, librosa, torch, torchaudio
- CosyVoice2 environment

**Optional (for full metrics):**
- whisper + jiwer (for WER)
- fastdtw (for better MCD alignment)  
- onnxruntime (for speaker similarity)

## Usage Examples

```bash
# Test specific components
python run_evaluation_pipeline.py --test-dataset
python run_evaluation_pipeline.py --test-synthesis
python run_evaluation_pipeline.py --test-metrics

# Run evaluation with custom config
python run_evaluation_pipeline.py --config my_config.yaml

# Quick test with 10 samples per split
# (edit eval_config.yaml: max_samples_per_split: 10)
python run_evaluation_pipeline.py
```
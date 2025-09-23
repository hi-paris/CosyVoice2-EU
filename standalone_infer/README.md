# CosyVoice2-EU

<div align="center">
  <img src="https://horstmann.tech/cosyvoice2-demo/cosyvoice2-logo-clear.png" alt="CosyVoice2-EU Logo" width="400"/>
</div>

Minimal, plug-and-play CosyVoice2 European inference CLI that downloads our model from Hugging Face and runs cross-lingual zero-shot voice cloning TTS. It bundles the required `cosyvoice` runtime and `matcha` module so you don't need the full upstream repo.

Currently supports Chinese, English, Japanese, Korean, Chinese dialects (Cantonese, Sichuanese, Shanghainese, Tianjinese, Wuhanese, etc.) from the original CosyVoice2, plus our newly added French and German support!

## Important Notes

- **Limited Training Data**: This model was fine-tuned on 1,000 hours of French and 1,000 hours of German data. Support and capabilities for these languages may still be limited compared to the original CosyVoice2 languages. Here, we ship a bilingual model for simplicity, but performance may vary between languages and when compared to monolingual model versions.
- **Prompt Support**: You can use prompts by putting your prompt text followed by `<|endofprompt|>` at the beginning of your text (e.g., `"Speak sadly. <|endofprompt|> Your actual text here"`). However, prompt support is currently limited and experimental.

## Quick Start (CLI)

1. **Install the package:**
   ```bash
   pip install cosyvoice2-eu
   ```

2. **Run French voice cloning:**
   ```bash
   cosy2-eu \
     --text "Salut ! Je vous présente CosyVoice 2, un système de synthèse vocale très avancé. Cette technologie permet de reproduire des voix de manière impressionnante." \
     --prompt french_speaker.wav \
     --out output_french.wav
   ```

3. **Run German voice cloning:**
   ```bash
   cosy2-eu \
     --text "Hallo! Ich stelle Ihnen CosyVoice 2 vor, ein sehr fortschrittliches Sprachsynthese-System. Diese Technologie kann Stimmen auf beeindruckende Weise reproduzieren." \
     --prompt german_speaker.wav \
     --out output_german.wav
   ```

4. **Use prompts for style control (experimental):**
   ```bash
   cosy2-eu \
     --text "Speak cheerfully. <|endofprompt|> Hallo! Wie geht es Ihnen heute? Ich hoffe, Sie haben einen wunderbaren Tag!" \
     --prompt german_speaker.wav \
     --out output_cheerful_german.wav
   ```

That's it! The first run will automatically download the model from Hugging Face. The model stays in memory between calls for faster subsequent inference.


## 🎯 Features

- **Easy Installation**: Simple `pip install cosyvoice2-eu` command
- **Cross-lingual Voice Cloning**: Clone voices across different languages
- **Multi-language Support**: 
  - **Original CosyVoice2**: Chinese, English, Japanese, Korean, Chinese dialects (Cantonese, Sichuanese, Shanghainese, Tianjinese, Wuhanese, etc.)
  - **European Extension**: French and German (fine-tuned on 1,500h each)
- **Model Caching**: Model stays in memory between calls for faster inference
- **Audio Concatenation**: Multiple audio segments are automatically concatenated into a single output file
- **Experimental Prompt Support**: Style control using `<|endofprompt|>` syntax (limited)
- **Bundled Runtime**: No need to install the full upstream CosyVoice2 repository
- **Hugging Face Integration**: Automatic model downloading from [Hugging Face](https://huggingface.co/Luka512/CosyVoice2-0.5B-EU)
- **Multiple LLM Backbones**: Support for different language model backbones (see below)
- **Text Frontend Disabled**: Text normalization is disabled by default for better multilingual support

## 📓 Interactive Usage (Python/Notebook)

You can keep the model in memory and call it multiple times without reloads:

```python
from cosyvoice2_eu import load
import torchaudio

# Load once (downloads on first use) and reuse
cosy = load()  # or: load(model_dir="~/.cache/cosyvoice2-eu", repo_id="Luka512/CosyVoice2-0.5B-EU")

# Full synthesis (returns a single waveform)
wav, sr = cosy.tts(
    text="Salut ! Ceci est une démonstration.",
    prompt="/path/to/french_ref.wav",
)
torchaudio.save("out_fr.wav", wav, sr)

# Streaming synthesis (yields chunks)
chunks = []
for chunk in cosy.stream(
    text="Hallo! Dies ist eine Streaming-Demonstration.",
    prompt="/path/to/german_ref.wav",
):
    chunks.append(chunk)
if chunks:
    import torch
    torchaudio.save("out_de_streamed.wav", torch.cat(chunks, dim=1), cosy.sample_rate)
```

## 🚀 Upcoming Features

**Multiple LLM Backbone Support** - Code is ready, models are currently training:
- **Qwen3 0.6B**: Lightweight model for efficient inference
- **EuroLLM 1.7B Instruct**: Specialized European language model
- **Mistral 7B v0.3**: Powerful multilingual capabilities

*Currently ships with the original CosyVoice2 "blankEN" backbone and our fine-tuned LM and flow models. New backbones will be available as separate model downloads once training is complete.*

## 📖 Model & Credits

This package uses our **CosyVoice2-0.5B-EU** model available at: 
🤗 [Luka512/CosyVoice2-0.5B-EU](https://huggingface.co/Luka512/CosyVoice2-0.5B-EU)

**Built on CosyVoice2**: This project builds upon the excellent
[CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) by FunAudioLLM (Apache 2.0),
adapted for European language support with cross‑lingual voice cloning capabilities.

## 📜 License & Attribution

- Package license: Apache License 2.0 (see `LICENSE`).
- Bundled upstream components (licenses included in distribution):
  - CosyVoice2 (FunAudioLLM) — Apache 2.0 • see `NOTICE` and `THIRD_PARTY_LICENSES/COSYVOICE_LICENSE`.
  - Matcha‑TTS (Shivam Mehta) — MIT • see `THIRD_PARTY_LICENSES/MATCHA_TTS_LICENSE`.
  - HiFi‑GAN (via Matcha‑TTS) — MIT • see `THIRD_PARTY_LICENSES/MATCHA_HIFIGAN_LICENSE`.

Original licenses and attributions are preserved. This package is not affiliated with
or endorsed by FunAudioLLM/Alibaba; trademarks and names belong to their owners.

## Installation

### From PyPI (Recommended)

```bash
pip install cosyvoice2-eu
```

### For enhanced English phonemization (optional):
```bash
pip install cosyvoice2-eu[piper]
```

**Note**: The `piper` optional dependency requires compilation tools and may fail in some environments (like Google Colab). The package will work without it, using the standard phonemizer as fallback.

If you are on Linux with GPU, ensure you install torch/torchaudio matching your CUDA and have `onnxruntime-gpu` available. If CPU-only, `onnxruntime` will be sufficient.

### Development Installation

```bash
cd standalone_infer
pip install -e .
```

## Usage

**French Example:**
```bash
cosy2-eu \
  --text "Salut ! Je vous présente CosyVoice 2, un système de synthèse vocale très avancé. Cette technologie permet de reproduire des voix de manière impressionnante." \
  --prompt french_speaker.wav \
  --out output_french.wav
```

**German Example:**
```bash
cosy2-eu \
  --text "Hallo! Ich stelle Ihnen CosyVoice 2 vor, ein sehr fortschrittliches Sprachsynthese-System. Diese Technologie kann Stimmen auf beeindruckende Weise reproduzieren." \
  --prompt german_speaker.wav \
  --out output_german.wav
```

**Prompt-based Style Control (Experimental):**
```bash
cosy2-eu \
  --text "Speak cheerfully. <|endofprompt|> Hallo! Wie geht es Ihnen heute? Ich hoffe, Sie haben einen wunderbaren Tag!" \
  --prompt german_speaker.wav \
  --out output_cheerful_german.wav
```

**English/Chinese/Japanese/Korean (Original CosyVoice2 languages):**
```bash
cosy2-eu \
  --text "Hello! This is CosyVoice 2, demonstrating cross-lingual voice cloning capabilities." \
  --prompt any_speaker.wav \
  --out output_english.wav
```

First run will download the model assets to `~/.cache/cosyvoice2-eu` (configurable via `--model-dir`). The model stays in memory between calls for faster subsequent inference.

**Advanced CLI options:** `--setting`, `--stream`, `--speed`, `--text-frontend` (enable text normalization), `--clear-cache` (reload model).



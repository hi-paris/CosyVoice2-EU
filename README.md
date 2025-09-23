<p align="center">
  <img src="https://horstmann.tech/cosyvoice2-demo/cosyvoice2-logo-clear.png" alt="CosyVoice2‑EU logo" width="260" />
</p>

<p align="center">
  <a href="https://pypi.org/project/cosyvoice2-eu/"><img src="https://img.shields.io/pypi/v/cosyvoice2-eu.svg?label=PyPI&color=%23009966" alt="PyPI"></a>
  <a href="https://huggingface.co/Luka512/CosyVoice2-0.5B-EU"><img src="https://img.shields.io/badge/HuggingFace-CosyVoice2--0.5B--EU-ffcc00?logo=huggingface" alt="HF"></a>
  <a href="https://horstmann.tech/cosyvoice2-demo/"><img src="https://img.shields.io/badge/Demo-Online-brightgreen" alt="Demo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="License"></a>
</p>

# CosyVoice2‑EU — FR/DE Zero‑Shot Voice Cloning

High‑quality French/German zero‑shot TTS built on CosyVoice2. Bilingual adaptation (FR+DE), streaming and non‑streaming synthesis, and a one‑command CLI via the companion PyPI package.

- PyPI: https://pypi.org/project/cosyvoice2-eu/
- Hugging Face: https://huggingface.co/Luka512/CosyVoice2-0.5B-EU
- Live demo: https://horstmann.tech/cosyvoice2-demo/

---

## Quickstart

Install the CLI:

```bash
pip install cosyvoice2-eu
```

(or if you use uv: `uv add cosyvoice2-eu --frozen`, then `uv sync`)

*This will install all necessary dependencies needed for inference with our CosyVoice2‑EU model.*

French example:

```bash
cosy2-eu \
  --text "Salut ! Je vous présente CosyVoice 2, un système de synthèse vocale très avancé." \
  --prompt path/to/french_ref.wav \
  --out out_fr.wav
```

German example:

```bash
cosy2-eu \
  --text "Hallo! Ich präsentiere CosyVoice 2 – ein fortschrittliches TTS‑System." \
  --prompt path/to/german_ref.wav \
  --out out_de.wav
```

Notes:
- First run downloads the model checkpoint and caches it locally.
- You can steer style via prompts, e.g. `"Speak cheerfully. <|endofprompt|> Hallo! Wie geht es Ihnen heute?"`.

### Python API (keep model in memory)

```python
from cosyvoice2_eu import load
import torchaudio

cosy = load()  # downloads on first use, then reuses the model
wav, sr = cosy.tts(
    text="Salut ! Ceci est une démo.",
    prompt="/path/to/french_ref.wav",
)
torchaudio.save("out.wav", wav, sr)
```

---

## What’s Inside (this repo)

- cosy_repo/: Local CosyVoice2 code, scripts and notebook for inference and utilities.
  - Notebook: `cosy_repo/inference_notebook.ipynb:1` (interactive local synthesis examples)
  - Script: `cosy_repo/run_inference.py:1` (command‑line inference with local checkpts)
- dataset: Different scripts to prepare datasets (LibriSpeech, EmoEnet) for training.
- evaluation/: Reproducible evaluation pipeline and configs used in our experiments.
  - Main config: `evaluation/eval_config.yaml:1`
  - Pipeline: `evaluation/run_evaluation_pipeline.py:1`
- standalone_infer/: Source of the PyPI package `cosyvoice2-eu` (packaging‑only).

---

## Benchmarking & Baselines

Here are the links to the baseline models we evaluated against:

- Coqui TTS (XTTS2): https://github.com/coqui-ai/TTS
- Fish‑Speech (OpenAudio S1 / related): https://github.com/fishaudio/fish-speech
- ElevenLabs (Flash V2.5): https://elevenlabs.io (proprietary, closed source)

Our evaluation pipeline supports running baselines independently; see `evaluation/run_baseline_evaluation.py:1` and per‑model example configs under `evaluation/` (e.g., `evaluation/eval_config_coqui.yaml:1`, `evaluation/eval_config_fishspeech.yaml:1`).

---

## Features

- Zero‑shot voice cloning for French and German
- Bilingual FR+DE adaptation on top of CosyVoice2
- Streaming and non‑streaming synthesis
- Simple CLI (`cosy2-eu`) and local inference scripts
- Interoperable, modular pipeline (text→semantic LM → flow decoder → HiFi‑GAN)

---

## Local Inference (from this repo)

If you’re working in this repository and want to run local inference with your own checkpoints, see:

- `cosy_repo/inference_notebook.ipynb:1`
- `cosy_repo/run_inference.py:1`

---

## Evaluation Pipeline

- Config: `evaluation/eval_config.yaml:1` controls language, budgets, metrics, and models.
- Run: `python evaluation/run_evaluation_pipeline.py --language fr --hours 100,250,500`
- Baselines: enable or run separately via the scripts and example configs under `evaluation/`.

---

## Model & Attribution

- Base models: FunAudioLLM/CosyVoice2‑0.5B, Qwen/Qwen3‑0.6B, utter‑project/EuroLLM‑1.7B‑Instruct, Mistral‑7B‑v0.3
- Built on CosyVoice2: https://github.com/FunAudioLLM/CosyVoice
- Hugging Face model: https://huggingface.co/Luka512/CosyVoice2-0.5B-EU

Please cite or acknowledge CosyVoice2 and the respective base LLMs when using this work.

---

## Data Sources

We prepare FR/DE training and evaluation splits from publicly available datasets. We do not
redistribute any third‑party data; please obtain them from the official sources and follow
their licenses/terms:

- Multilingual LibriSpeech (MLS): https://huggingface.co/datasets/facebook/multilingual_librispeech
- LAION “LAION’s Got Talent” annotations (used for expressive data):
  https://huggingface.co/datasets/laion/laions_got_talent_enhanced_flash_annotations_and_long_captions
- M‑AILABS Speech Dataset (OOD evaluation): https://github.com/imdatceleste/m-ailabs-dataset
- Mozilla Common Voice (prompt references): https://commonvoice.mozilla.org

See `NOTICE` for a concise overview.

---

## License & Notices

- Root license: Apache License 2.0 (see `LICENSE`).
- CosyVoice2 code in `cosy_repo/` is from FunAudioLLM/CosyVoice and remains under Apache‑2.0 (see `cosy_repo/LICENSE`).
- The packaging project in `standalone_infer/` is released under Apache‑2.0 (see `standalone_infer/LICENSE`).

See `NOTICE` for a concise overview of third‑party components and their licenses.

---

## Links

- PyPI (CLI): https://pypi.org/project/cosyvoice2-eu/
- Hugging Face (model): https://huggingface.co/Luka512/CosyVoice2-0.5B-EU
- Live demo: https://horstmann.tech/cosyvoice2-demo/
- Upstream CosyVoice2: https://github.com/FunAudioLLM/CosyVoice
- Coqui TTS (XTTS2): https://github.com/coqui-ai/TTS
- Fish‑Speech: https://github.com/fishaudio/fish-speech
- ElevenLabs: https://elevenlabs.io

---

## Citation

If you use CosyVoice2‑EU in research or products, please cite:

- CosyVoice2‑EU (this work): preprint forthcoming. See `CITATION.cff:1` for metadata.
- Upstream CosyVoice2: FunAudioLLM/CosyVoice — please also cite their paper and repo.

BibTeX entries (will be updated soon!)

```bibtex
@misc{horstmann2025cosyvoice2eu,
  title        = {CosyVoice2-EU: Europeanized CosyVoice2 for French and German Zero-Shot TTS},
  author       = {Horstmann, Tim Luka and Ould Ouali, Nassima and Arous, Mohamed Amine and Hussain Sani, Awais and Moulines, Eric},
  year         = {2025},
  note         = {Preprint in preparation},
  howpublished = {\url{https://horstmann.tech/cosyvoice2-demo/}}
}

@misc{du2024cosyvoice2,
  title        = {CosyVoice2},
  author       = {Du, et al.},
  year         = {2024},
  howpublished = {\url{https://github.com/FunAudioLLM/CosyVoice}}
}
```

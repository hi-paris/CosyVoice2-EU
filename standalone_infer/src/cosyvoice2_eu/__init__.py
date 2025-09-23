"""CosyVoice2‑EU importable API for interactive use.

Example:

    from cosyvoice2_eu import load
    cosy = load()  # downloads & caches on first use
    wav, sr = cosy.tts(
        text="Bonjour, ceci est une démo.",
        prompt="/path/to/french_ref.wav",
    )

    # Or stream chunks
    for chunk in cosy.stream(text="...", prompt="..."):
        pass
"""

from typing import Iterator, Tuple, Optional
import os
import torch
from huggingface_hub import snapshot_download

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

__all__ = [
    "__version__",
    "Cosy2EU",
    "load",
]
__version__ = "0.2.8"


class Cosy2EU:
    """Lightweight wrapper around CosyVoice2 for interactive inference."""

    def __init__(self, model: CosyVoice2):
        self._model = model

    @property
    def sample_rate(self) -> int:
        return getattr(self._model, "sample_rate", 24000)

    def tts(
        self,
        text: str,
        prompt: str,
        *,
        speed: float = 1.0,
        text_frontend: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """Synthesize full audio for given text and prompt.

        - prompt: path to a wav file (>=16 kHz). It is resampled to 16 kHz.
        - returns: (waveform [1, T] torch.Tensor, sample_rate)
        """
        prompt_16k = load_wav(prompt, 16000)
        segments = []
        for _, out in enumerate(
            self._model.inference_cross_lingual(
                text,
                prompt_16k,
                stream=False,
                speed=speed,
                text_frontend=text_frontend,
            )
        ):
            segments.append(out["tts_speech"])  # [1, t]
        if len(segments) == 1:
            wav = segments[0]
        else:
            wav = torch.cat(segments, dim=1)
        return wav, self.sample_rate

    def stream(
        self,
        text: str,
        prompt: str,
        *,
        speed: float = 1.0,
        text_frontend: bool = False,
    ) -> Iterator[torch.Tensor]:
        """Yield audio chunks ([1, t] torch.Tensor) for streaming playback."""
        prompt_16k = load_wav(prompt, 16000)
        for _, out in enumerate(
            self._model.inference_cross_lingual(
                text,
                prompt_16k,
                stream=True,
                speed=speed,
                text_frontend=text_frontend,
            )
        ):
            yield out["tts_speech"]


def load(
    *,
    model_dir: Optional[str] = None,
    repo_id: str = "Luka512/CosyVoice2-0.5B-EU",
    download: bool = True,
    setting: str = "llm_flow_hifigan",
    llm_run_id: str = "latest",
    flow_run_id: str = "latest",
    hifigan_run_id: str = "latest",
    final: Optional[bool] = None,
    backbone: str = "blanken",
) -> Cosy2EU:
    """Load CosyVoice2‑EU once and reuse for multiple in‑memory calls.

    Returns a Cosy2EU wrapper with `tts` and `stream` methods.
    """
    model_dir = model_dir or os.path.expanduser("~/.cache/cosyvoice2-eu")
    if download:
        snapshot_download(repo_id=repo_id, local_dir=model_dir)
    model = CosyVoice2(
        model_dir,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=False,
        setting=setting,
        llm_run_id=llm_run_id,
        flow_run_id=flow_run_id,
        hifigan_run_id=hifigan_run_id,
        final=(True if final is None else final),
        backbone=backbone,
    )
    return Cosy2EU(model)


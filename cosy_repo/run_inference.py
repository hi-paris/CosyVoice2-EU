import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio, os, argparse, torch
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

# ---- Cache keys ----
@dataclass(frozen=True)
class ModelKey:
    model_dir: str
    setting: str
    llm_run_id: str
    flow_run_id: str
    hifigan_run_id: str
    final: bool
    backbone: Optional[str]

# Model loader is itself cached; changing any field => new instance
@lru_cache(maxsize=1)  # keep last used model; bump if want multiple resident
def _load_model(key: ModelKey):
    m = CosyVoice2(
        key.model_dir,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=False,
        setting=key.setting,
        llm_run_id=key.llm_run_id,
        flow_run_id=key.flow_run_id,
        hifigan_run_id=key.hifigan_run_id,
        final=key.final,
        backbone=key.backbone,
    )
    if hasattr(m, "eval"):
        m.eval()
    return m

# Optional: cache prompt audio to avoid re-reading same file
@lru_cache(maxsize=8)
def _load_prompt_wav(path: str, sr: int = 16000):
    return load_wav(path, sr)

def clear_model_cache():
    """Manual nuke (e.g., after CUDA OOM or device change)."""
    _load_model.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_inference(
    text: str,
    prompt_wav: str,
    output_wav: Optional[str],
    model_dir: str,
    setting: str = 'llm_flow_hifigan',
    llm_run_id: str = 'latest',
    flow_run_id: str = 'latest',
    hifigan_run_id: str = 'latest',
    final: bool = True,
    stream: bool = False,
    speed: float = 1.0,
    text_frontend: bool = True,
    backbone: str = None,
):
    if output_wav is None:
        output_wav = f'audio_output/cl-{setting}.wav'
    os.makedirs(os.path.dirname(output_wav) or '.', exist_ok=True)

    key = ModelKey(
        model_dir=model_dir,
        setting=setting,
        llm_run_id=llm_run_id,
        flow_run_id=flow_run_id,
        hifigan_run_id=hifigan_run_id,
        final=final,
        backbone=backbone,
    )
    model = _load_model(key)
    prompt_speech_16k = _load_prompt_wav(prompt_wav, 16000)

    outputs = []
    with torch.no_grad():
        for _, chunk in enumerate(
            model.inference_cross_lingual(
                text,
                prompt_speech_16k,
                stream=stream,
                speed=speed,
                text_frontend=text_frontend,
            )
        ):
            outputs.append(chunk['tts_speech'])

    if outputs:
        concat_audio = torch.cat(outputs, dim=1)
        torchaudio.save(output_wav, concat_audio, model.sample_rate)
        print(f"Output saved to {output_wav}")
    else:
        print("No audio generated.")

    return output_wav

def main():
    parser = argparse.ArgumentParser(description='CosyVoice2 EU Inference (cross-lingual cloning)')
    parser.add_argument('--text', type=str, required=False,
                        default="Bonjour, je m'appelle Emmanuel et je travaille dans une entreprise de technologie à Paris. Aujourd'hui, nous allons explorer les capacités de synthèse vocale en français avec CosyVoice2.")
    parser.add_argument('--prompt', type=str, required=False, default='prompt_audio/macron.wav', help='Path to a ≥16kHz prompt wav')
    parser.add_argument('--out', type=str, required=False, help='Defaults to audio_output/cl-{{setting}}.wav')
    parser.add_argument('--model-dir', type=str, required=False, default='pretrained_models/CosyVoice2-0.5B-EU')
    parser.add_argument('--repo-id', type=str, required=False, default='Luka512/CosyVoice2-0.5B-EU', help='HF repo to auto-download into --model-dir when --no-hf is not set')
    parser.add_argument('--no-hf', action='store_true', help='Do not download from HF; assume --model-dir already exists')
    parser.add_argument('--setting', type=str, default='llm_flow_hifigan', help='See cosyvoice2 settings: original|llm|flow|hifigan|llm_flow|llm_hifigan|flow_hifigan|llm_flow_hifigan')
    parser.add_argument('--llm-run-id', type=str, default='latest')
    parser.add_argument('--flow-run-id', type=str, default='latest')
    parser.add_argument('--hifigan-run-id', type=str, default='latest')
    parser.add_argument('--final', action='store_true', help='Use final checkpoints (llm.pt/flow.pt/hift.pt)')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--no-text-frontend', action='store_true', help='Disable text normalization frontend')
    parser.add_argument('--backbone', type=str, default=None, help='LLM backbone type (e.g., \"blanken\", \"hf:Qwen/Qwen2.5-0.5B\", \"hf:utter-project/EuroLLM-1.7B-Instruct\"). If not specified, will auto-detect from model directory.')
    args = parser.parse_args()

    model_dir = args.model_dir
    final = args.final or (not args.no_hf)  # default to final=True when using HF snapshot

    if not args.no_hf:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.repo_id, local_dir=model_dir)

    run_inference(
        text=args.text,
        prompt_wav=args.prompt,
        output_wav=args.out,
        model_dir=model_dir,
        setting=args.setting,
        llm_run_id=args.llm_run_id,
        flow_run_id=args.flow_run_id,
        hifigan_run_id=args.hifigan_run_id,
        final=final,
        stream=args.stream,
        speed=args.speed,
        text_frontend=not args.no_text_frontend,
        backbone=args.backbone,
    )

if __name__ == '__main__':
    main()



# '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
#                 '[breath]', '<strong>', '</strong>', '[noise]',
#                 '[laughter]', '[cough]', '[clucking]', '[accent]',
#                 '[quick_breath]',
#                 "<laughter>", "</laughter>",
#                 "[hissing]", "[sigh]", "[vocalized-noise]",
#                 "[lipsmack]", "[mn]"


# for i, j in enumerate(cosyvoice.inference_zero_shot(
#     #'<|en|>Hey, my name is LeBron James, I am an NBA player and a 4 times world champion, some refer to me as King James.'
#     "Bertrand Périer est avocat au Conseil d’État et à la Cour de cassation à Paris. Spécialiste de l’éloquence, il enseigne l’art oratoire à Sciences Po et participe au programme Eloquentia. Il défend l’idée que la parole est un outil d’émancipation pour tous."
#     , ""
#     , prompt_speech_16k
#     , stream=False
#     , text_frontend=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)




# # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# # zero_shot usage
# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # save zero_shot spk for future usage
# assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# cosyvoice.save_spkinfo()

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # bistream usage, you can use generator as input, this is useful when using text llm model as input
# # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
# def text_generator():
#     yield '收到好友从远方寄来的生日礼物，'
#     yield '那份意外的惊喜与深深的祝福'
#     yield '让我心中充满了甜蜜的快乐，'
#     yield '笑容如花儿般绽放。'
# for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
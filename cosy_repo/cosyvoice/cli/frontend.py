# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Generator, Callable
import json
import onnxruntime
import torch
import numpy as np
import whisper
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import re
import inflect
from pathlib import Path

# ---------------- Optional, robust TN for EN/DE/FR ----------------
# pip install nemo_text_processing
try:
    from nemo_text_processing.text_normalization.normalize import Normalizer as NemoNormalizer
    _HAS_NEMO_TN = True
except Exception:
    _HAS_NEMO_TN = False

# ---------------- Optional language ID (high quality) -------------
# pip install lingua-language-detector
try:
    from lingua import Language, LanguageDetectorBuilder
    _HAS_LINGUA = True
except Exception:
    _HAS_LINGUA = False

# ---------------- Optional DE helpers from your utils -------------
try:
    from cosyvoice.utils.frontend_utils import (
        contains_german as _contains_german_utils,
        expand_abbreviations_german as _expand_abbreviations_german_utils,
        spell_out_number_german as _spell_out_number_german_utils,
        replace_symbols_german as _replace_symbols_german_utils,
    )
    _HAS_DE_UTILS = True
except Exception:
    _HAS_DE_UTILS = False

# ---------------- Fallbacks for German if utils are absent --------
try:
    from num2words import num2words as _num2words
    _HAS_NUM2WORDS = True
except Exception:
    _HAS_NUM2WORDS = False

def _fallback_contains_german(text: str) -> bool:
    # broadened signals (covers sentences like "Genau genommen seit 2020.")
    if re.search(r"[äöüÄÖÜß]", text):
        return True
    if re.search(r"\b("
                 r"und|oder|nicht|mit|ist|ein|eine|der|die|das|zum|beispiel|bzw|"
                 r"genau|genommen|seit|schon|bereits|heute|gestern|morgen|wird|wurden?|kann|können|deutsch|spr[eä]che?"
                 r")\b", text, re.IGNORECASE):
        return True
    return False

def _fallback_expand_abbreviations_german(text: str) -> str:
    ABR = {
        r"\bz\.?\s?B\.?\b": "zum Beispiel",
        r"\bu\.?\s?a\.?\b": "unter anderem",
        r"\bbzw\.?\b": "beziehungsweise",
        r"\bca\.?\b": "circa",
        r"\bd\.?\s?h\.?\b": "das heißt",
        r"\binsb\.?\b": "insbesondere",
        r"\bNr\.?\b": "Nummer",
        r"\bS\.?\b": "Seite",
    }
    for pat, rep in ABR.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text

def _fallback_replace_symbols_german(text: str) -> str:
    text = text.replace("€", " Euro ")
    text = text.replace("%", " Prozent ")
    text = re.sub(r"\bkm/?h\b", " Kilometer pro Stunde ", text, flags=re.IGNORECASE)
    text = text.replace("&", " und ")
    text = text.replace("@", " at ")
    text = text.replace("§", " Paragraph ")
    text = text.replace("°C", " Grad Celsius ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _fallback_spell_out_number_german(text: str) -> str:
    def _ord_repl(m):
        n = int(m.group(1))
        if _HAS_NUM2WORDS:
            try:
                return _num2words(n, lang="de", to="ordinal")
            except Exception:
                return f"{n}."
        return f"{n}."
    text = re.sub(r"\b(\d+)\.(?=\s|$)", _ord_repl, text)

    def _dec_repl(m):
        s = m.group(0)
        s = s.replace(".", "").replace(" ", "")
        intp, frac = s.split(",", 1)
        if _HAS_NUM2WORDS:
            try:
                left = _num2words(int(intp), lang="de")
            except Exception:
                left = intp
        else:
            left = intp
        frac_spelled = " ".join({
            "0":"null","1":"eins","2":"zwei","3":"drei","4":"vier",
            "5":"fünf","6":"sechs","7":"sieben","8":"acht","9":"neun"
        }.get(ch, ch) for ch in frac)
        return f"{left} Komma {frac_spelled}"
    text = re.sub(r"\b\d{1,3}(?:[.\s]\d{3})*,\d+\b", _dec_repl, text)

    def _int_repl(m):
        s = m.group(0).replace(".", "").replace(" ", "")
        if _HAS_NUM2WORDS:
            try:
                return _num2words(int(s), lang="de")
            except Exception:
                return s
        return s
    text = re.sub(r"\b\d{1,3}(?:[.\s]\d{3})+\b", _int_repl, text)
    text = re.sub(r"\b\d+\b", _int_repl, text)
    return text

# ---------------- Your existing utils -----------------------------
try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False

from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    contains_french,
    replace_blank,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    spell_out_number_french,
    replace_symbols_french,
    expand_abbreviations_french,
    split_paragraph,
    is_only_punctuation,
)

def contains_german(text: str) -> bool:
    if _HAS_DE_UTILS:
        return _contains_german_utils(text)
    return _fallback_contains_german(text)


class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 allowed_special: str = 'all'):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1

        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model,
            sess_options=option,
            providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"]
        )

        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
        else:
            self.spk2info = {}

        self.allowed_special = allowed_special
        self.use_ttsfrd = use_ttsfrd

        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            import os as _os
            _default_resource = f'{ROOT_DIR}/../../pretrained_models/CosyVoice-ttsfrd/resource'
            _resource_path = _os.environ.get('COSY_TTSFRD_RESOURCE', _default_resource)
            assert self.frd.initialize(_resource_path) is True, \
                f'failed to initialize ttsfrd resource at {_resource_path}'
            self.frd.set_lang_type('pinyinvg')
            self.en_tn_model = None
            self.zh_tn_model = None
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=True)
            self.en_tn_model = EnNormalizer()

        # inflect for EN last-resort numbers
        self.inflect_parser = inflect.engine()

        # NeMo WFST normalizers for en/de/fr (lazy; cache to disk so we don't rebuild every process)
        self.nemo_norm = {}
        self.nemo_cache_dir = os.environ.get(
            "NEMO_TN_CACHE",
            os.path.join(Path.home(), ".cache", "cosyvoice", "nemo_tn")
        )
        if _HAS_NEMO_TN:
            os.makedirs(self.nemo_cache_dir, exist_ok=True)


        # Optional Lingua LID for EN/DE/FR (we bypass it for zh)
        self.lid = None
        if _HAS_LINGUA:
            try:
                langs = [Language.ENGLISH, Language.GERMAN, Language.FRENCH]
                self.lid = LanguageDetectorBuilder.from_languages(*langs).build()
            except Exception:
                self.lid = None

    # ---------------- Low-level extractors (unchanged) ---------------

    def _extract_text_token(self, text):
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will return _extract_text_token_generator!')
            return self._extract_text_token_generator(text), torch.tensor([0], dtype=torch.int32).to(self.device)
        else:
            text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i: i + 1]

    def _extract_speech_token(self, speech):
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(
            None,
            {
                self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                self.speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
            },
        )[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(
            None, {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
        )[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    # ---------------- Multilingual sentence-level TN -----------------

    @staticmethod
    def _split_sentences(text: str):
        return [s.strip() for s in re.split(r'(?<=[\.\?\!\u2026\u3002\uff01\uff1f])\s+', text) if s.strip()]

    def _detect_lang(self, s: str) -> str:
        # Always short-circuit Chinese
        if contains_chinese(s):
            return "zh"

        # Prefer Lingua if available
        if self.lid is not None:
            try:
                lang = self.lid.detect_language_of(s)
                if str(lang) == "German":
                    return "de"
                if str(lang) == "French":
                    return "fr"
                if str(lang) == "English":
                    return "en"
            except Exception:
                pass

        # Heuristics (fast and reliable enough for short sents)
        if contains_french(s):
            return "fr"
        if contains_german(s):
            return "de"
        return "en"

    def _get_nemo_normalizer(self, lang: str):
        if not _HAS_NEMO_TN:
            return None
        if lang in self.nemo_norm:
            return self.nemo_norm[lang]
        try:
            norm = NemoNormalizer(
                lang=lang,
                input_case="cased",
                deterministic=True,
                cache_dir=self.nemo_cache_dir,
                overwrite_cache=False
            )
            self.nemo_norm[lang] = norm
            return norm
        except Exception:
            return None


    def _normalize_sentence(self, s: str, lang: str) -> str:
        s = s.replace("\n", " ").strip()

        # Chinese (ttsfrd or WeTextProcessing)
        if lang == "zh":
            if self.use_ttsfrd:
                try:
                    zh_texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(s))["sentences"]]
                    return "".join(zh_texts)
                except Exception:
                    pass
            if hasattr(self, "zh_tn_model") and self.zh_tn_model is not None:
                s = self.zh_tn_model.normalize(s)
                s = s.replace("\n", "")
                s = replace_blank(s)
                s = replace_corner_mark(s)
                s = s.replace(".", "。").replace(" - ", "，")
                s = remove_bracket(s)
                s = re.sub(r'[  \t]+', ' ', s)  # collapse spaces
                s = re.sub(r'[，,、]+$', '。', s)
                return s
            return s

        # Use NeMo WFST normalizers for en/de/fr when available
        if lang in ("en", "de", "fr"):
            norm = self._get_nemo_normalizer(lang)
            if norm is not None:
                # Guard: if the sentence ENDS with "<digits>." treat the dot as EOS, not ordinal marker.
                eos_punct = ""
                if re.search(r"\d\.$", s):
                    eos_punct = "."
                    s = s[:-1]  # drop the dot before normalization

                try:
                    out = norm.normalize(s)
                except Exception:
                    out = s  # fall through to light fallbacks below

                if eos_punct:
                    out = out.rstrip() + eos_punct

                return out

        # Fallback FR
        if lang == "fr":
            s = expand_abbreviations_french(s)
            s = spell_out_number_french(s)
            s = replace_symbols_french(s)
            s = remove_bracket(s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        # Fallback DE
        if lang == "de":
            if _HAS_DE_UTILS:
                s = _expand_abbreviations_german_utils(s)
                s = _spell_out_number_german_utils(s)
                s = _replace_symbols_german_utils(s)
            else:
                s = _fallback_expand_abbreviations_german(s)
                s = _fallback_spell_out_number_german(s)
                s = _fallback_replace_symbols_german(s)
            s = remove_bracket(s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        # Fallback EN
        if hasattr(self, "en_tn_model") and self.en_tn_model is not None:
            try:
                return self.en_tn_model.normalize(s)
            except Exception:
                pass
        try:
            s = spell_out_number(s, self.inflect_parser)
        except Exception:
            pass
        return s

    def text_normalize(
        self,
        text,
        split=True,
        text_frontend=True,
        multilingual=True,
        pack_mode: str = "sentence",          # "sentence" | "paragraph"
        target_token_len: int = 512           # ~6–7 s at CV2 (25 tps, ratio=2 -> 50 fps)
    ):
        """
        Robust multilingual TN with optional paragraph packing.
        pack_mode="sentence": current behavior.
        pack_mode="paragraph": TN per sentence, then re-pack across sentences by token budget.
        """
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will skip text_normalize!')
            return [text]
        if text_frontend is False or text == '':
            return [text] if split else text

        text = text.strip()
        if not text:
            return [""] if split else ""

        # 1) sentence-level TN (keep as-is)
        normalized_sents = []
        sents = self._split_sentences(text) if multilingual else [text]
        for sent in sents:
            lang = self._detect_lang(sent)
            normalized_sents.append(self._normalize_sentence(sent, lang))

        # 2) packing
        if pack_mode == "paragraph":
            # Join all normalized sentences, keep spaces; then split ONLY by token budget.
            big = " ".join(normalized_sents).strip()
            chunk_lang = "en"  # token-length driven; language-specific rules are already done
            max_n = int(target_token_len)
            min_n = max(1, int(0.75 * target_token_len))
            merge_n = max(1, int(0.25 * target_token_len))
            chunks = list(split_paragraph(
                big,
                partial(self.tokenizer.encode, allowed_special=self.allowed_special),
                chunk_lang,
                token_max_n=max_n, token_min_n=min_n, merge_len=merge_n, comma_split=False
            ))
            texts = [t for t in chunks if not is_only_punctuation(t)]
            return texts if split else " ".join(texts)

        # pack_mode == "sentence": existing behavior (token-bound per sentence)
        segments = []
        for sent in normalized_sents:
            # choose chunking profile only for splitting granularity; TN was already language-specific
            segs = list(split_paragraph(
                sent,
                partial(self.tokenizer.encode, allowed_special=self.allowed_special),
                "en",  # safe default for Latin; zh was normalized with Chinese punctuation already
                token_max_n=80, token_min_n=60, merge_len=20, comma_split=False
            ))
            segments.extend(segs)

        texts = [t for t in segments if not is_only_punctuation(t)]
        return texts if split else " ".join(texts)


    # ---------------- Remaining frontends (unchanged) ----------------

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]['embedding']
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        if zero_shot_spk_id == '':
            prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
            prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
            speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
            if resample_rate == 24000:
                # cosyvoice2, force speech_feat % speech_token = 2
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
                speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
            embedding = self._extract_spk_embedding(prompt_speech_16k)
            model_input = {'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                           'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                           'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                           'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                           'llm_embedding': embedding, 'flow_embedding': embedding}
        else:
            model_input = self.spk2info[zero_shot_spk_id]
        model_input['text'] = tts_text_token
        model_input['text_len'] = tts_text_token_len
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k, resample_rate, zero_shot_spk_id)
        # in cross lingual mode, we remove prompt in llm
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input['llm_embedding']
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input

    def frontend_instruct2(self, tts_text, instruct_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        model_input = self.frontend_zero_shot(tts_text, instruct_text + '<|endofprompt|>', prompt_speech_16k, resample_rate, zero_shot_spk_id)
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_speech_16k)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k)
        model_input = {'source_speech_token': source_speech_token, 'source_speech_token_len': source_speech_token_len,
                       'flow_prompt_speech_token': prompt_speech_token, 'flow_prompt_speech_token_len': prompt_speech_token_len,
                       'prompt_speech_feat': prompt_speech_feat, 'prompt_speech_feat_len': prompt_speech_feat_len,
                       'flow_embedding': embedding}
        return model_input

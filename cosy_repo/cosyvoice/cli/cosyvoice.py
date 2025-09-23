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
import os
import time
from typing import Generator
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type


class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) != CosyVoice2Model, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id):
        assert zero_shot_spk_id != '', 'do not use empty zero_shot_spk_id'
        model_input = self.frontend.frontend_zero_shot('', prompt_text, prompt_speech_16k, self.sample_rate, '')
        del model_input['text']
        del model_input['text_len']
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        torch.save(self.frontend.spk2info, '{}/spk2info.pt'.format(self.model_dir))

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoiceModel), 'inference_instruct is only implemented for CosyVoice!'
        if self.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False
        , fp16=False, trt_concurrent=1, setting="original"
        , llm_run_id=None
        , flow_run_id=None
        , hifigan_run_id=None
        , final=False
        , backbone=None
        ):
        """

        Args:
            model_dir: model directory, should contain cosyvoice2.yaml and pretrained models
            load_jit: whether to load JIT model
            load_trt: whether to load TensorRT model
            load_vllm: whether to load vLLM model
            fp16: whether to use fp16 model
            trt_concurrent: number of concurrent threads for TensorRT inference
            setting: 'original' for original/unchanged CosyVoice2, 'llm' for updated LLM only, 'flow' for updated flow only, 'hifigan' for updated HiFi-GAN only, 'llm_flow_hifigan' for updated LLM, flow and HiFi-GAN, can be combined like 'llm_flow', 'llm_hifigan', 'flow_hifigan'
            llm_run_id: llm run id to specify the model to be used, if None, will use original model
            flow_run_id: flow run id to specify the model to be used, if None, will use original model
            hifigan_run_id: hifigan run id to specify the model to be used, if None, will use original model
            final: whether to use final version of the model --> will use llm.pt and flow.pt instead of llm-original.pt and flow-original.pt
            backbone: LLM backbone type/path (e.g., 'blanken', 'hf:Qwen/Qwen2.5-0.5B', 'hf:utter-project/EuroLLM-1.7B-Instruct'). 
                     If None, will auto-detect from model directory or use default behavior.
        """
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            overrides = {}
            
            # Handle backbone parameter to set qwen_pretrain_path appropriately
            qwen_pretrain_path = None
            
            if backbone is not None:
                # Explicit backbone provided - parse it
                if backbone == "blanken":
                    blank_path = os.path.join(model_dir, 'CosyVoice-BlankEN')
                    if os.path.exists(blank_path):
                        qwen_pretrain_path = blank_path
                elif backbone.startswith("hf:"):
                    qwen_pretrain_path = backbone[3:]  # Remove 'hf:' prefix
                elif backbone.startswith("local:"):
                    qwen_pretrain_path = backbone[6:]  # Remove 'local:' prefix
                else:
                    qwen_pretrain_path = backbone  # Direct path
            else:
                # # Auto-detect: Try to load backbone info from model directory
                # backbone_info_path = os.path.join(model_dir, 'backbone_info.txt')
                # if os.path.exists(backbone_info_path):
                #     try:
                #         with open(backbone_info_path, 'r') as bf:
                #             saved_backbone = bf.read().strip()
                #         logging.info(f'Auto-detected backbone from {backbone_info_path}: {saved_backbone}')
                        
                #         if saved_backbone == "blanken":
                #             blank_path = os.path.join(model_dir, 'CosyVoice-BlankEN')
                #             if os.path.exists(blank_path):
                #                 qwen_pretrain_path = blank_path
                #         elif saved_backbone.startswith("hf:"):
                #             qwen_pretrain_path = saved_backbone[3:]
                #         elif saved_backbone.startswith("local:"):
                #             qwen_pretrain_path = saved_backbone[6:]
                #         else:
                #             qwen_pretrain_path = saved_backbone
                #     except Exception as e:
                #         logging.warning(f'Failed to read backbone info: {e}')
                
                # Fallback: Original BlankEN detection logic (maintain backward compatibility)
                blank_path = os.path.join(model_dir, 'CosyVoice-BlankEN')
                if os.path.exists(blank_path) and not os.environ.get("COSYV2_IGNORE_BLANKEN"):
                    qwen_pretrain_path = blank_path
            
            if qwen_pretrain_path:
                overrides['qwen_pretrain_path'] = qwen_pretrain_path
                logging.info(f'Using LLM backbone: {qwen_pretrain_path}')
            
            configs = load_hyperpyyaml(f, overrides=overrides)
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        # Determine which checkpoints to load
        if final:
            tokens = {"llm", "flow", "hifigan"}
        else:
            if setting == "original":
                tokens = set()
            else:
                tokens = set(setting.split("_"))
                allowed = {"llm", "flow", "hifigan"}
                invalid = tokens - allowed
                if invalid:
                    raise ValueError(f'setting should be one of "original", "llm", "flow", "hifigan", "llm_flow", "llm_hifigan", "flow_hifigan", "llm_flow_hifigan", but got {setting}')
        # Map each component to its run_id and file key
        components = [
            ("llm", llm_run_id),
            ("flow", flow_run_id),
            ("hift", hifigan_run_id),
        ]
        chosen = {}
        for key, run_id in components:
            token = key if key != "hift" else "hifigan"
            if key in {"llm", "flow", "hift"} and (final or (token in tokens and run_id is not None)):
                suffix = "" if final else f"-{run_id}"
            else:
                suffix = "-original"
            chosen[key] = f"{model_dir}/{key}{suffix}.pt"
        chosen_llm, chosen_flow, chosen_hift = chosen["llm"], chosen["flow"], chosen["hift"]
        
        print(f"Loading CosyVoice2 with\n\tLLM: {chosen_llm}\n\tFlow: {chosen_flow}\n\tHiFT: {chosen_hift}")
        self.model.load(chosen_llm, chosen_flow, chosen_hift)
        
        if load_vllm:
            self.model.load_vllm('{}/vllm'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2!')

    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

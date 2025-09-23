# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
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
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = [
    'Cross-lingual Cloning / Clonage multilingue',
]
instruct_dict = {
    'Cross-lingual Cloning / Clonage multilingue': '1. Upload or record a voice prompt (max 30s)\n1. Téléchargez ou enregistrez un échantillon vocal (30s max)\n2. Click Generate Audio\n2. Cliquez sur Générer l\'audio',
}
stream_mode_list = [('No / Non', False), ('Yes / Oui', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, prompt_wav_upload, prompt_wav_record, seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    # Validation
    if prompt_wav is None:
        gr.Warning('Voice prompt is required / Échantillon vocal requis')
        yield (cosyvoice.sample_rate, default_data)
        return
    
    if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
        gr.Warning('Voice prompt sample rate {} is too low (minimum {}) / Fréquence d\'échantillonnage {} trop faible (minimum {})'.format(
            torchaudio.info(prompt_wav).sample_rate, prompt_sr, torchaudio.info(prompt_wav).sample_rate, prompt_sr))
        yield (cosyvoice.sample_rate, default_data)
        return

    logging.info('Starting cross-lingual inference for French synthesis')
    prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
    set_all_random_seed(seed)
    
    for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed, text_frontend=False):
        yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### CosyVoice2 French Fine-tuned Model / Modèle CosyVoice2 adapté au français")
        gr.Markdown("Cross-lingual voice cloning for French text synthesis / Clonage vocal multilingue pour la synthèse française")
        
        tts_text = gr.Textbox(
            label="French text to synthesize / Texte français à synthétiser", 
            lines=3, 
            value="Bonjour, je suis Bertrand Perrier et vous écoutez ma masterclass. Je suis avocat et j'aime une baguette."
        )
        
        with gr.Row():
            mode_checkbox_group = gr.Radio(
                choices=inference_mode_list, 
                label='Mode', 
                value=inference_mode_list[0]
            )
            instruction_text = gr.Text(
                label="Instructions", 
                value=instruct_dict[inference_mode_list[0]], 
                scale=0.6
            )
            
        with gr.Row():
            stream = gr.Radio(
                choices=stream_mode_list, 
                label='Streaming / Continu', 
                value=stream_mode_list[0][1]
            )
            speed = gr.Number(
                value=1, 
                label="Speed / Vitesse", 
                minimum=0.5, 
                maximum=2.0, 
                step=0.1
            )
            with gr.Column(scale=0.3):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(
                sources='upload', 
                type='filepath', 
                label='Upload voice prompt (≥16 kHz) / Téléchargez un échantillon vocal (≥16 kHz)'
            )
            prompt_wav_record = gr.Audio(
                sources='microphone', 
                type='filepath', 
                label='Record voice prompt / Enregistrez un échantillon vocal'
            )

        generate_button = gr.Button("Generate Audio / Générer l'audio", variant="primary")
        audio_output = gr.Audio(label="Synthesized Audio / Audio synthétisé", autoplay=True, streaming=True)

        # Event handlers
        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            generate_audio,
            inputs=[tts_text, mode_checkbox_group, prompt_wav_upload, prompt_wav_record, seed, stream, speed],
            outputs=[audio_output]
        )
        mode_checkbox_group.change(
            fn=change_instruction, 
            inputs=[mode_checkbox_group], 
            outputs=[instruction_text]
        )
    
    demo.queue(max_size=4, default_concurrency_limit=2)
    
    print(f"\nFrench TTS WebUI starting...")
    print(f"Local access: http://localhost:{args.port}")
    if args.share:
        print(f"Public HTTPS link will be generated for full functionality...")
    print(f"Press Ctrl+C to stop the server\n")
    
    demo.launch(
        server_name='0.0.0.0', 
        server_port=args.port,
        share=args.share,  # Creates a secure public gradio.live link
        show_error=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B')
    parser.add_argument('--llm_run_id', type=str, default='original') # 373115
    parser.add_argument('--flow_run_id', type=str, default='original') # 372090
    parser.add_argument('--share', action='store_true', help='Create a public HTTPS link for full functionality (recommended for remote access)')
    args = parser.parse_args()
    
    # Initialize model with your fine-tuned parameters
    cosyvoice = CosyVoice2(
        args.model_dir,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=False,
        setting='all',
        llm_run_id=args.llm_run_id,
        flow_run_id=args.flow_run_id
    )

    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()

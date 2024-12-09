from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
import os
import ipdb
import gradio as gr
import mdtex2html
from model.openllama import OpenLLAMAPEFTModel
import torch
import json

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}
model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()
print(f'[!] init the model over ...')

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def re_predict(
    input, 
    image_path, 
    audio_path, 
    video_path, 
    thermal_path, 
    chatbot, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    modality_cache, 
):
    # drop the latest query and answers and generate again
    q, a = history.pop()
    chatbot.pop()
    return predict(q, image_path, audio_path, video_path, thermal_path, chatbot, max_length, top_p, temperature, history, modality_cache)


def predict(
    input, 
    image_path, 
    audio_path, 
    video_path, 
    thermal_path, 
    chatbot, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    modality_cache, 
):
    if image_path is None and audio_path is None and video_path is None and thermal_path is None:
        return [(input, "There is no input data provided! Please upload your data and start the conversation.")]
    else:
        print(f'[!] image path: {image_path}\n[!] audio path: {audio_path}\n[!] video path: {video_path}\n[!] thermal path: {thermal_path}')

    # prepare the prompt
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        'thermal_paths': [thermal_path] if thermal_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })
    chatbot.append((parse_text(input), parse_text(response)))
    history.append((input, response))
    return chatbot, history, modality_cache


def reset_user_input():
    return gr.update(value='')

def reset_dialog():
    return [], []

def reset_state():
    return None, None, None, None, [], [], []


# Define Gradio app layout
with gr.Blocks() as demo:
    gr.HTML("<h1 align='center'>PandaGPT</h1>")

    with gr.Row():
        # Upload components for image, audio, video, and thermal image inputs
        image_path = gr.Image(type="filepath", label="Image")
        audio_path = gr.Audio(type="filepath", label="Audio")
        video_path = gr.Video(type='file', label="Video")
        thermal_path = gr.Image(type="filepath", label="Thermal Image")

    # Chatbot interface
    chatbot = gr.Chatbot().style(height=300)

    # User input section
    with gr.Row():
        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=3).style(container=False)
        submitBtn = gr.Button("Submit", variant="primary")
        resubmitBtn = gr.Button("Resubmit", variant="primary")
        emptyBtn = gr.Button("Clear History")

    # Settings section for max length, top p, and temperature
    with gr.Row():
        max_length = gr.Slider(minimum=0, maximum=400, value=256, label="Maximum Length")
        top_p = gr.Slider(minimum=0, maximum=1, value=0.01, step=0.01, label="Top P")
        temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.01, label="Temperature")

    # States to keep track of chatbot history and modality cache
    history = gr.State([])
    modality_cache = gr.State([])

    # Define interactions
    submitBtn.click(
        fn=predict, 
        inputs=[user_input, image_path, audio_path, video_path, thermal_path, chatbot, max_length, top_p, temperature, history, modality_cache], 
        outputs=[chatbot, history, modality_cache],
        show_progress=True
    )

    resubmitBtn.click(
        fn=re_predict, 
        inputs=[user_input, image_path, audio_path, video_path, thermal_path, chatbot, max_length, top_p, temperature, history, modality_cache], 
        outputs=[chatbot, history, modality_cache],
        show_progress=True
    )

    # Additional interactions
    submitBtn.click(fn=reset_user_input, inputs=[], outputs=[user_input])
    emptyBtn.click(fn=reset_state, inputs=[], outputs=[image_path, audio_path, video_path, thermal_path, chatbot, history, modality_cache])

# Launch the Gradio app
demo.launch(share=True)

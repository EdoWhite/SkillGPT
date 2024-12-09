import argparse
import torch
from transformers import AutoModel, AutoTokenizer
from model.openllama import OpenLLAMAPEFTModel
import json

# Initialize the model with the necessary arguments
def initialize_model():
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
    return model

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

def generate_response(model, input_text, image_path=None, audio_path=None, video_path=None, thermal_path=None, max_length=256, top_p=0.9, temperature=1.0):
    # Prepare the prompt
    prompt_text = input_text

    # Generate response using the model
    response = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        'thermal_paths': [thermal_path] if thermal_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': ([])  # Assuming no modality embeddings in the CLI contex
    })

    return response

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Interact with the VLM model from the command line.")
    parser.add_argument("--input", type=str, required=True, help="Input text for the model.")
    parser.add_argument("--image_path", type=str, help="Path to an image file (optional).")
    parser.add_argument("--audio_path", type=str, help="Path to an audio file (optional).")
    parser.add_argument("--video_path", type=str, help="Path to a video file (optional).")
    parser.add_argument("--thermal_path", type=str, help="Path to a thermal image file (optional).")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length of the output text.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling value.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature value for sampling.")

    args = parser.parse_args()

    # Load and initialize the model
    model = initialize_model()
    print("[!] Model initialized successfully.")

    # Generate a response based on the inputs
    response = generate_response(
        model, 
        input_text=args.input, 
        image_path=args.image_path, 
        audio_path=args.audio_path, 
        video_path=args.video_path, 
        thermal_path=args.thermal_path, 
        max_length=args.max_length, 
        top_p=args.top_p, 
        temperature=args.temperature
    )

    # Print the response
    print(f"Model Response: {parse_text(response)}")

if __name__ == "__main__":
    main()

import torch
import random
import base64
import io
import requests
from PIL import Image
from .utils import *
from .image_utils import *

import torch
import base64
import io
import requests
from PIL import Image

class OpenAI_API_Call:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["gpt-4o", "gpt-4", "gpt-3.5-turbo", "gpt-4.1-mini"],),
                "system_instructions": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "api_key": ("STRING", {"multiline": False}),
                "seed" : ("INT", {"forceInput":True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "call_openai"
    CATEGORY = PACK_NAME

    def call_openai(self, image, model, system_instructions, user_prompt, temperature, frequency_penalty, api_key, seed:int):
        # Convert image tensor to base64
        image_np = (image[0].clamp(0, 1).cpu().numpy() * 255).astype('uint8')
        image_pil = Image.fromarray(image_np)
        buffered = io.BytesIO()
        image_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Build API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "messages": [
                {"role": "system", "content": system_instructions},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return (result['choices'][0]['message']['content'],)
        else:
            return (f"Error: {response.status_code}, {response.text}",)


NODE_CLASS_MAPPINGS = {
    "OpenAI_API_Call": OpenAI_API_Call
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAI_API_Call": PACK_NAME + "chatgpt"

}
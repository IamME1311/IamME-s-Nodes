import torch
import google.generativeai as genai
from PIL import Image
from .utils import *
from .image_utils import *

class GeminiVisionV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-preview-05-20", "gemini-2.5-flash-preview-04-17-thinking","gemini-2.5-pro"],
                    {"default": "gemini-2.5-flash-preview-05-20"}
                ),
                "seed": ("INT", {"forceInput": True}),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0, "max": 2, "step": 0.1, "display": "slider"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.85, "min": 0, "max": 1, "step": 0.05, "display": "slider"
                }),
                "top_k": ("INT", {
                    "default": 40, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "api_key": ("STRING", {"default": ""}),
                "system_instruction": ("STRING", {
                    "default": "You are a professional AI assistant that writes elegant, highly realistic photography prompts for luxury watch product shots.",
                    "multiline": True
                }),
                "prompt": ("STRING", {
                    "default": "Write a background scene for a luxury watch photo.",
                    "multiline": False
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = PACK_NAME
    FUNCTION = 'execute'

    def resize_image_to_800(self, image, max_size=800):
        """
        Resize PIL image to maximum 800px on the longest side while maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            max_size: Maximum size for the longest side (default: 800)
        
        Returns:
            PIL Image object resized to max 800px
        """
        if image is None:
            return None
        
        # Get original dimensions
        width, height = image.size
        
        # If image is already smaller than max_size, return as-is
        if max(width, height) <= max_size:
            return image
        
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        # Resize the image using high-quality resampling
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return resized_image

    def execute(self,
                model_name: str,
                seed: int,
                temperature: float,
                top_p: float,
                top_k: int,
                system_instruction: str,
                prompt: str,
                api_key: str,
                image: torch.Tensor = None,
                image2: torch.Tensor = None
                ) -> tuple[str]:
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Error configuring Gemini model: {e}")

        llm = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        )

        tensor_images = [image, image2]
        pil_images = tensor_batch2pil(tensor_images)
        resized_images = [self.resize_image_to_800(img) for img in pil_images if img is not None]

        log_to_console("Resized images to max 800px and converted to PIL format", 20)
        response = llm.generate_content([*resized_images, prompt])

        return (response.text,)


NODE_CLASS_MAPPINGS = {"Gem_Chat": GeminiVisionV2}
NODE_DISPLAY_NAME_MAPPINGS = {"Gem_chat": PACK_NAME + "Gem_chat"}
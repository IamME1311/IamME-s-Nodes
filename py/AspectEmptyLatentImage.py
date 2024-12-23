import math
import torch
import comfy
from .utils import *

class AspectEmptyLatentImage:
    def __init__(self) -> None:
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s) -> dict[str:tuple[str, dict]]:
        return {
            "required": { 
                "width": ("INT", {"default":1024, "min":16, "max":MAX_RESOLUTION, "step":8}),
                "height": ("INT", {"default":1024, "min":16, "max":MAX_RESOLUTION, "step":8}),
                "aspect_ratio": (ASPECT_CHOICES, {"default": "None"}),
                "model_type":(["SD1.5", "SDXL"],),
                "aspect_width_override" : ("INT",{"default":0, "min":0}),
                "aspect_height_override" : ("INT",{"default":0, "min":0}),
                "width_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("LATENT", "INT", "INT", IMAGE_DATA["type"])
    RETURN_NAMES = ("samples","width","height", IMAGE_DATA["name"])
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "execute"

    CATEGORY = PACK_NAME
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def execute(self,
                    width:int,
                    height:int,
                    model_type:str,
                    aspect_ratio:int,
                    aspect_width_override:int,
                    aspect_height_override:int,
                    width_override:int,
                    batch_size:int=1
                    ) -> dict:
        if aspect_ratio!="None":
            if aspect_ratio in ASPECT_CHOICES[2:]:
                aspect_width_override, aspect_height_override = parser(aspect_ratio)
            if width_override > 0:
                width = width_override - (width_override%8)
                height = int((aspect_height_override * width) / aspect_width_override)
                height = height - (height%8)
            else:
                total_pixels = {
                    "SD1.5" : 512 * 512,
                    "SDXL" : 1024 * 1024
                }
                pixels = total_pixels.get(model_type, 0)

                aspect_ratio_value = aspect_width_override / aspect_height_override

                width = int(math.sqrt(pixels * aspect_ratio_value))
                height = int(pixels / width)


        else: # normal empty latent
            width = int(width - (width%8))
            height = int(height - (height%8))

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        image_data = {"width":width, "height":height}

        return {"ui":{
                    "text": [f"{height}x{width}"]
                },
                "result":
                    ({"samples":latent}, width, height, image_data)
            }
    

NODE_CLASS_MAPPINGS = {"AspectEmptyLatentImage" : AspectEmptyLatentImage,}

NODE_DISPLAY_NAME_MAPPINGS = {"AspectEmptyLatentImage" : PACK_NAME + " AspectEmptyLatent",}
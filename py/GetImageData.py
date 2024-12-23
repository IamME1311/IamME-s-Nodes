import math
import torch
from .utils import *
class GetImageData:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "Image" : ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", IMAGE_DATA["type"])
    RETURN_NAMES = ("Image", "Width", "Height", "Aspect Ratio", IMAGE_DATA["name"])
    FUNCTION = "execute"
    CATEGORY = PACK_NAME

    def execute(self, Image:torch.Tensor) -> dict:
        width = Image.shape[2]
        height = Image.shape[1]
        aspect_ratio_str = f"{int(width / math.gcd(width, height))}:{int(height / math.gcd(width, height))}"

        if width > height:
            orientation = "Landscape"
        elif height > width:
            orientation = "Portrait"
        else:
            orientation = "Square"
        image_data = {"width":width, "height":height, "aspect_ratio_str":aspect_ratio_str, "orientation":orientation}
        return {
            "ui" : {
                "text" : [f"{aspect_ratio_str}"],
            },
            "result" : 
            (Image, width, height, aspect_ratio_str, image_data),
        }
    
NODE_CLASS_MAPPINGS = {"GetImageData": GetImageData,}
NODE_DISPLAY_NAME_MAPPINGS = {"GetImageData": PACK_NAME + " GetImageData",}
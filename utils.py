import json
import os
import numpy as np
from PIL import Image
import torch



MAX_RESOLUTION = 16384
ASPECT_CHOICES = ["None","custom",
                    "1:1 (Perfect Square)",
                    "2:3 (Classic Portrait)", "3:4 (Golden Ratio)", "3:5 (Elegant Vertical)", "4:5 (Artistic Frame)", "5:7 (Balanced Portrait)", "5:8 (Tall Portrait)",
                    "7:9 (Modern Portrait)", "9:16 (Slim Vertical)", "9:19 (Tall Slim)", "9:21 (Ultra Tall)", "9:32 (Skyline)",
                    "3:2 (Golden Landscape)", "4:3 (Classic Landscape)", "5:3 (Wide Horizon)", "5:4 (Balanced Frame)", "7:5 (Elegant Landscape)", "8:5 (Cinematic View)",
                    "9:7 (Artful Horizon)", "16:9 (Panorama)", "19:9 (Cinematic Ultrawide)", "21:9 (Epic Ultrawide)", "32:9 (Extreme Ultrawide)"
                ]

DEFAULT_SYS_PROMPT = ""


def json_loader(file_name:str) -> dict:
    cwd_name = os.path.dirname(__file__)
    path_to_asset_file = os.path.join(cwd_name, f"assets/{file_name}.json")
    with open(path_to_asset_file, "r") as f:
        asset_data = json.load(f)
    return asset_data

def apply_attention(text:str, weight:float) -> str:
    weight = float(np.round(weight, 2))
    return f"({text}:{weight})"

def tensor_to_image(tensor) -> Image:
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def image_to_tensor(image : Image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def parser(aspect : str) -> int:
    aspect = list(map(int,aspect.split()[0].split(":")))
    return aspect

random_opt = "Randomize 🎲"
option_dict = json_loader("FacePromptMaker")


# thanks to pythongossss..
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")
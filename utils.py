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

IMAGE_DATA = {"type":"image_data", "name":"image data"}

BUS_DATA = {"type":"bus", "name":"bus"}


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
    return Image.fromarray(np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def image_to_tensor(image : Image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def parser(aspect : str) -> int:
    aspect = list(map(int,aspect.split()[0].split(":")))
    return aspect

random_opt = "Randomize 🎲"
option_dict = json_loader("FacePromptMaker")


def gamma_correction_tensor(image, gamma):
    gamma_inv = 1.0 / gamma
    return image.pow(gamma_inv)

def contrast_adjustment_tensor(image, contrast):
    r, g, b = image.unbind(-1)

    # Using Adobe RGB luminance weights.
    luminance_image = 0.33 * r + 0.71 * g + 0.06 * b
    luminance_mean = torch.mean(luminance_image.unsqueeze(-1))

    # Blend original with mean luminance using contrast factor as blend ratio.
    contrasted = image * contrast + (1.0 - contrast) * luminance_mean
    return torch.clamp(contrasted, 0.0, 1.0)


def exposure_adjustment_tensor(image, exposure):
    return image * (2.0**exposure)


def offset_adjustment_tensor(image, offset):
    return image + offset


def hsv_adjustment(image: torch.Tensor, hue, saturation, value):
    img = tensor_to_image(image)
    hsv_image = img.convert("HSV")

    h, s, v = hsv_image.split()

    h = h.point(lambda x: (x + hue * 255) % 256)
    s = s.point(lambda x: int(x * saturation))
    v = v.point(lambda x: int(x * value))

    hsv_image = Image.merge("HSV", (h, s, v))
    rgb_image = hsv_image.convert("RGB")
    return image_to_tensor(rgb_image)

def color_balance(image:torch.tensor, shadows:list, midtones:list, highlights:list,
                  shadow_center:float=0.15, midtone_center:float=0.5, highlight_center:float=0.8,
                  shadow_max:float=0.1, midtone_max:float=0.3, highlight_max:float=0.2,
                  preserve_luminosity:bool=False) -> Image:

    img = image
    # Create a copy of the img tensor
    img_copy = img.clone()

    # Calculate the original luminance if preserve_luminosity is True
    if preserve_luminosity:
        original_luminance = 0.2126 * img_copy[..., 0] + 0.7152 * img_copy[..., 1] + 0.0722 * img_copy[..., 2]

    # Define the adjustment curves
    def adjust(x, center, value, max_adjustment):
        # Scale the adjustment value
        value = value * max_adjustment

        # Define control points
        points = torch.tensor([[0, 0], [center, center + value], [1, 1]])

        # Create cubic spline
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(points[:, 0], points[:, 1])

        # Apply the cubic spline to the color channel
        return torch.clamp(torch.from_numpy(cs(x)), 0, 1)

    # Apply the adjustments to each color channel
    # shadows, midtones, highlights are lists of length 3 (for R, G, B channels) with values between -1 and 1
    for i, (s, m, h) in enumerate(zip(shadows, midtones, highlights)):
        img_copy[..., i] = adjust(img_copy[..., i], shadow_center, s, shadow_max)
        img_copy[..., i] = adjust(img_copy[..., i], midtone_center, m, midtone_max)
        img_copy[..., i] = adjust(img_copy[..., i], highlight_center, h, highlight_max)

    # If preserve_luminosity is True, adjust the RGB values to match the original luminance
    if preserve_luminosity:
        current_luminance = 0.2126 * img_copy[..., 0] + 0.7152 * img_copy[..., 1] + 0.0722 * img_copy[..., 2]
        img_copy *= (original_luminance / current_luminance).unsqueeze(-1)

    return img_copy

def RGB2RGBA(image:torch.tensor, mask:torch.tensor) -> torch.tensor:
    image = tensor_to_image(image)
    mask = tensor_to_image(mask)
    (R, G, B) = image.convert('RGB').split()
    return image_to_tensor(Image.merge('RGBA', (R, G, B, mask.convert('L'))))

# thanks to pythongossss..
class AnyType(str):

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")
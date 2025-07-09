import base64
import io
from PIL import Image, ImageEnhance, ImageChops, ImageFilter
import scipy.ndimage
import cv2
import torch
import numpy as np
from .blendmodes import *



#_________from layer_style nodes________
def tensor2pil(tensor:torch.Tensor) -> Image:
    """
    converts a pytorch tensor to PIL image format.
    """
    return Image.fromarray(np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image : Image):
    """
    converts a PIL image to pytorch tensor format.
    """
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil2cv2(pil_img:Image) -> np.array:
    """
    converts a PIL image to cv2 format.
    """
    np_img_array = np.asarray(pil_img)
    return cv2.cvtColor(np_img_array, cv2.COLOR_RGB2BGR)

def cv22pil(cv2_img:np.ndarray) -> Image:
    """
    converts a cv2 image to PIL image format.
    """
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

def image2mask(image:Image) -> torch.Tensor:
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])

def expand_mask(mask:torch.Tensor, grow:int, blur:int) -> torch.Tensor:
    # grow
    c = 0
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in growmask:
        output = m.numpy()
        for _ in range(abs(grow)):
            if grow < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    # blur
    for idx, tensor in enumerate(out):
        pil_image = tensor2pil(tensor.cpu().detach())
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur))
        out[idx] = pil2tensor(pil_image)
    ret_mask = torch.cat(out, dim=0)
    return ret_mask


def color_balance(image:Image, shadows:list, midtones:list, highlights:list,
                  shadow_center:float=0.15, midtone_center:float=0.5, highlight_center:float=0.8,
                  shadow_max:float=0.1, midtone_max:float=0.3, highlight_max:float=0.2,
                  preserve_luminosity:bool=False) -> Image:
    """
    color balances a PIL image with the following params:
    shadows : 
    midtones : 
    highlights :
    shadow_center :
    midtone_center :
    highlight_center :
    shadow_max :
    midtone_max :
    highlight_max :
    preserve_luminosity : 
    """

    img = pil2tensor(image)
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

def image_hue_offset(image:Image, offset:int) -> Image:
    image = image.convert('L')
    width = image.width
    height = image.height
    ret_image = Image.new('L', size=(width, height), color='black')
    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            _pixel = pixel + offset
            if _pixel > 255:
                _pixel -= 256
            if _pixel < 0:
                _pixel += 256
            ret_image.putpixel((x, y), _pixel)
    return ret_image

def image_gray_offset(image:Image, offset:int) -> Image:
    image = image.convert('L')
    width = image.width
    height = image.height
    ret_image = Image.new('L', size=(width, height), color='black')
    for x in range(width):
        for y in range(height):
                pixel = image.getpixel((x, y))
                _pixel = pixel + offset
                if _pixel > 255:
                    _pixel = 255
                if _pixel < 0:
                    _pixel = 0
                ret_image.putpixel((x, y), _pixel)
    return ret_image

def image_channel_merge(channels:tuple, mode = 'RGB' ) -> Image:
    channel1 = channels[0].convert('L')
    channel2 = channels[1].convert('L')
    channel3 = channels[2].convert('L')
    channel4 = Image.new('L', size=channel1.size, color='white')
    if mode == 'RGBA':
        if len(channels) > 3:
            channel4 = channels[3].convert('L')
        ret_image = Image.merge('RGBA',[channel1, channel2, channel3, channel4])
    elif mode == 'RGB':
        ret_image = Image.merge('RGB', [channel1, channel2, channel3])
    elif mode == 'YCbCr':
        ret_image = Image.merge('YCbCr', [channel1, channel2, channel3]).convert('RGB')
    elif mode == 'LAB':
        ret_image = Image.merge('LAB', [channel1, channel2, channel3]).convert('RGB')
    elif mode == 'HSV':
        ret_image = Image.merge('HSV', [channel1, channel2, channel3]).convert('RGB')
    return ret_image


def RGB2RGBA(image:torch.tensor, mask:torch.tensor) -> torch.tensor:
    image = tensor2pil(image)
    mask = tensor2pil(mask)
    (R, G, B) = image.convert('RGB').split()
    return pil2tensor(Image.merge('RGBA', (R, G, B, mask.convert('L'))))


def gamma_trans(image:Image, gamma:float) -> Image:
    cv2_image = pil2cv2(image)
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    _corrected = cv2.LUT(cv2_image,gamma_table)
    return cv22pil(_corrected)

def chop_image_v2(background_image:Image, layer_image:Image, blend_mode:str, opacity:int) -> Image:

    backdrop_prepped = np.asfarray(background_image.convert('RGBA'))
    source_prepped = np.asfarray(layer_image.convert('RGBA'))
    blended_np = BLEND_MODES[blend_mode](backdrop_prepped, source_prepped, opacity / 100)

    # final_tensor = (torch.from_numpy(blended_np / 255)).unsqueeze(0)
    # return tensor2pil(_tensor)

    return Image.fromarray(np.uint8(blended_np)).convert('RGB')


############################################################################
def tensor2base64(image:torch.Tensor) -> bytes:
    img = tensor2pil(image)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_bytes

def tensor_batch2pil(images:list) -> list:
    batch_size = len(images) # shape is in format of [batch_size, height, width, channel_count]
    pil_images = []
    for index in range(batch_size):
        if images[index]==None:
            continue
        pil_images.append(tensor2pil(images[index]))
    return pil_images

def tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
    """Converts a ComfyUI-style tensor (B, H, W, C) [0,1] to an OpenCV-style BGR uint8 image (H, W, C) [0,255]."""
    img_tensor = tensor[0]
    img_np = np.clip(255. * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr

def cv2_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Converts an OpenCV-style BGR uint8 image (H, W, C) [0,255] to a ComfyUI-style tensor (B, H, W, C) [0,1]."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_float).unsqueeze(0)
    return tensor
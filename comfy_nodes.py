import comfy
import math
from .blendmodesfile import *
import copy


MAX_RESOLUTION = 16384

class AspectEmptyLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "width": ("INT", {"default":1024, "min":16, "max":MAX_RESOLUTION, "step":8}),
                "height": ("INT", {"default":1024, "min":16, "max":MAX_RESOLUTION, "step":8}),
                "aspect_control" : ("BOOLEAN", {"default":False, "tooltip":"Set to 'True' if you want to select size based on a particular aspect ratio."}),
                "model_type":(["SD1.5", "SDXL"],),
                "aspect_width_ratio" : ("INT",{"default":9, "min":0}),
                "aspect_height_ratio" : ("INT",{"default":16, "min":0}),
                "width_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("samples","width","height",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "aspect_latent_gen"

    CATEGORY = "IamME"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def aspect_latent_gen(self, width, height, model_type, aspect_control, aspect_width_ratio, aspect_height_ratio, width_override, batch_size=1):
        if aspect_control==True:
            if width_override > 0:
                width = width_override - (width_override%8)
                height = int((aspect_height_ratio * width) / aspect_width_ratio)
                height = height - (height%8)
            else:
                total_pixels = {
                    "SD1.5" : 512 * 512,
                    "SDXL" : 1024 * 1024
                }
                pixels = total_pixels.get(model_type, 0)

                aspect_ratio_value = aspect_width_ratio / aspect_height_ratio

                width = int(math.sqrt(pixels * aspect_ratio_value))
                height = int(pixels / width)

        else:
            width = int(width - (width%8))
            height = int(height - (height%8))
    
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return {"ui":{
                    "text": [f"{height}x{width}"]
                }, 
                "result":
                    ({"samples":latent}, width, height)
            }


class LiveTextEditor:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                    "text":("STRING", {"forceInput": True}),
                    # "clip" : ("CLIP")
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    # RETURN_TYPES = ("CONDITIONING",)
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "TextEditor"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "IamME"

    def TextEditor(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}

    # def TextEditor(self, text, clip):
        
    #     text_final = text
        
    #     # encoding
    #     tokens = clip.tokenize(text_final)
    #     output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    #     cond = output.pop("cond")
    #     return ([[cond, output]], )


class ImageBlendLivePreview:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        mirror_mode = ['None', 'horizontal', 'vertical']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        chop_mode_v2 = list(BLEND_MODES.keys())
        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode_v2,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "x_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "y_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "mirror": (mirror_mode,),  # 镜像翻转
                "scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "aspect_ratio": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "rotate": ("FLOAT", {"default": 0, "min": -999999, "max": 999999, "step": 0.01}),
                "transform_method": (method_mode,),
                "anti_aliasing": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'blend_image'
    CATEGORY = 'IamME'

    def tensor2pil(t_image: torch.Tensor)  -> Image:
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    def pil2tensor(image:Image) -> torch.Tensor:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    def log(message:str, message_type:str='info'):
        name = 'IamME'

        if message_type == 'error':
            message = '\033[1;41m' + message + '\033[m'
        elif message_type == 'warning':
            message = '\033[1;31m' + message + '\033[m'
        elif message_type == 'finish':
            message = '\033[1;32m' + message + '\033[m'
        else:
            message = '\033[1;33m' + message + '\033[m'
        print(f"# IamME: {name} -> {message}")
    def image2mask(image:Image) -> torch.Tensor:
        _image = image.convert('RGBA')
        alpha = _image.split() [0]
        bg = Image.new("L", _image.size)
        _image = Image.merge('RGBA', (bg, bg, bg, alpha))
        ret_mask = torch.tensor([ImageBlendLivePreview.pil2tensor(_image)[0, :, :, 3].tolist()])
        return ret_mask
    def chop_image_v2(background_image:Image, layer_image:Image, blend_mode:str, opacity:int) -> Image:

        backdrop_prepped = np.asfarray(background_image.convert('RGBA'))
        source_prepped = np.asfarray(layer_image.convert('RGBA'))
        blended_np = BLEND_MODES[blend_mode](backdrop_prepped, source_prepped, opacity / 100)

        # final_tensor = (torch.from_numpy(blended_np / 255)).unsqueeze(0)
        # return tensor2pil(_tensor)
        return Image.fromarray(np.uint8(blended_np)).convert('RGB')
    def RGB2RGBA(image:Image, mask:Image) -> Image:
        (R, G, B) = image.convert('RGB').split()
        return Image.merge('RGBA', (R, G, B, mask.convert('L')))
    def image_rotate_extend_with_alpha(image:Image, angle:float, alpha:Image=None, method:str="lanczos", SSAA:int=0) -> tuple:
        _image = ImageBlendLivePreview.__rotate_expand(image.convert('RGB'), angle, SSAA, method)
        if angle is not None:
            _alpha = ImageBlendLivePreview.__rotate_expand(alpha.convert('RGB'), angle, SSAA, method)
            ret_image = ImageBlendLivePreview.RGB2RGBA(_image, _alpha)
        else:
            ret_image = _image
        return (_image, _alpha.convert('L'), ret_image)
    def __rotate_expand(image:Image, angle:float, SSAA:int=0, method:str="lanczos") -> Image:
        images = ImageBlendLivePreview.pil2tensor(image)
        expand = "true"
        height, width = images[0, :, :, 0].shape

        def rotate_tensor(tensor):
            resize_sampler = Image.LANCZOS
            rotate_sampler = Image.BICUBIC
            if method == "bicubic":
                resize_sampler = Image.BICUBIC
                rotate_sampler = Image.BICUBIC
            elif method == "hamming":
                resize_sampler = Image.HAMMING
                rotate_sampler = Image.BILINEAR
            elif method == "bilinear":
                resize_sampler = Image.BILINEAR
                rotate_sampler = Image.BILINEAR
            elif method == "box":
                resize_sampler = Image.BOX
                rotate_sampler = Image.NEAREST
            elif method == "nearest":
                resize_sampler = Image.NEAREST
                rotate_sampler = Image.NEAREST
            img = ImageBlendLivePreview.tensor2pil(tensor)
            if SSAA > 1:
                img_us_scaled = img.resize((width * SSAA, height * SSAA), resize_sampler)
                img_rotated = img_us_scaled.rotate(angle, rotate_sampler, expand == "true", fillcolor=(0, 0, 0, 0))
                img_down_scaled = img_rotated.resize((img_rotated.width // SSAA, img_rotated.height // SSAA), resize_sampler)
                result = ImageBlendLivePreview.pil2tensor(img_down_scaled)
            else:
                img_rotated = img.rotate(angle, rotate_sampler, expand == "true", fillcolor=(0, 0, 0, 0))
                result = ImageBlendLivePreview.pil2tensor(img_rotated)
            return result

        if angle == 0.0 or angle == 360.0:
            return ImageBlendLivePreview.tensor2pil(images)
        else:
            rotated_tensor = torch.stack([rotate_tensor(images[i]) for i in range(len(images))])
            return ImageBlendLivePreview.tensor2pil(rotated_tensor).convert('RGB')

    def blend_image(self, background_image, layer_image,
                            invert_mask, blend_mode, opacity,
                            x_percent, y_percent,
                            mirror, scale, aspect_ratio, rotate,
                            transform_method, anti_aliasing,
                            layer_mask=None
                            ):
        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
        ret_masks = []
        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in layer_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = ImageBlendLivePreview.tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(ImageBlendLivePreview.tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        max_batch = max(len(b_images), len(l_images), len(l_masks))
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            # preprocess
            _canvas = ImageBlendLivePreview.tensor2pil(background_image).convert('RGB')
            _layer = ImageBlendLivePreview.tensor2pil(layer_image)

            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                ImageBlendLivePreview.log(f"Warning: ImageBlendLivePreview mask mismatch, dropped!", message_type='warning')

            orig_layer_width = _layer.width
            orig_layer_height = _layer.height
            _mask = _mask.convert("RGB")

            target_layer_width = int(orig_layer_width * scale)
            target_layer_height = int(orig_layer_height * scale * aspect_ratio)

            # mirror
            if mirror == 'horizontal':
                _layer = _layer.transpose(Image.FLIP_LEFT_RIGHT)
                _mask = _mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif mirror == 'vertical':
                _layer = _layer.transpose(Image.FLIP_TOP_BOTTOM)
                _mask = _mask.transpose(Image.FLIP_TOP_BOTTOM)

            # scale
            _layer = _layer.resize((target_layer_width, target_layer_height))
            _mask = _mask.resize((target_layer_width, target_layer_height))
            # rotate
            _layer, _mask, _ = ImageBlendLivePreview.image_rotate_extend_with_alpha(_layer, rotate, _mask, transform_method, anti_aliasing)

            # 处理位置
            x = int(_canvas.width * x_percent / 100 - _layer.width / 2)
            y = int(_canvas.height * y_percent / 100 - _layer.height / 2)

            # composit layer
            _comp = copy.copy(_canvas)
            _compmask = Image.new("RGB", _comp.size, color='black')
            _comp.paste(_layer, (x, y))
            _compmask.paste(_mask, (x, y))
            _compmask = _compmask.convert('L')
            _comp = ImageBlendLivePreview.chop_image_v2(_canvas, _comp, blend_mode, opacity)

            # composition background
            _canvas.paste(_comp, mask=_compmask)

            ret_images.append(ImageBlendLivePreview.pil2tensor(_canvas))
            ret_masks.append(ImageBlendLivePreview.image2mask(_compmask))

        ImageBlendLivePreview.log(f"ImageBlendLivePreview Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)















NODE_CLASS_MAPPINGS = {
    "AspectEmptyLatentImage" : AspectEmptyLatentImage,
    "LiveTextEditor": LiveTextEditor, 
    "ImageBlendLivePreview":ImageBlendLivePreview
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectEmptyLatentImage" : "AspectEmptyLatentImage",
    "LiveTextEditor" : "LiveTextEditor",
    "ImageBlendLivePreview":"ImageBlendLivePreview"
}
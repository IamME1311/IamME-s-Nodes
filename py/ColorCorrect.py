import torch
from .utils import *
from .image_utils import *
class ColorCorrect:

    @classmethod
    def INPUT_TYPES(s) -> dict[str:tuple[str, dict]]:
        return {
            "required" : {
                "images" : ("IMAGE",),
                "gamma": (
                    "FLOAT",
                    {"default": 1, "min": 0.1, "max": 10, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1, "min": 0.0, "max": 3, "step": 0.01},
                ),
                "exposure": (
                    "INT",
                    {"default": 0, "min": -100, "max": 100, "step": 1},
                ),
                "temperature": (
                    "FLOAT", 
                    {"default": 0, "min": -100, "max": 100, "step": 1},
                ),
                "hue": (
                    "INT",
                    {"default": 0, "min": -255, "max": 255, "step": 1},
                ),
                "saturation": (
                    "INT",
                    {"default": 0, "min": -255, "max": 255, "step": 1},
                ),
                "value": (
                    "INT",
                    {"default": 0, "min": -255, "max": 255, "step": 1},
                ),
                "cyan_red": (
                    "FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}
                ),
                "magenta_green": (
                    "FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}
                ),
                "yellow_blue": (
                    "FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}
                ),
            },
            "optional": {
                "mask": ("MASK",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = PACK_NAME
    FUNCTION = "execute"

    def execute(self,
                    images : torch.Tensor,
                    gamma : float=1,
                    contrast : float=1,
                    exposure : int=0,
                    hue : int=0,
                    saturation : int=0,
                    value : int=0,
                    cyan_red : float=0,
                    magenta_green : float=0,
                    yellow_blue : float=0,
                    temperature : float=0,
                    mask : torch.Tensor | None = None
                ) -> tuple[torch.Tensor]:


        l_images = []
        l_masks = []
        return_images = []

        for l in images:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))

        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        if len(l_images) != len(l_masks):
            raise ValueError("Number of images and masks is not the same, aborting!!")

        for  index, image in enumerate(images):
            log_to_console(f"index:{index}, no of images:{len(images)}")
            mask = l_masks[index]
            # image = l_images[index]
            # preserving original image for later comparisions
            original_image = tensor2pil(image)

            # apply temperature [takes in tensor][gives out tensor]
            if temperature != 0:
                log_to_console("in temperature") # for debugging
                temperature /= -100
                # result = torch.zeros_like(i)

                tensor_image = image.numpy()
                modified_image = Image.fromarray((tensor_image * 255).astype(np.uint8))
                modified_image = np.array(modified_image).astype(np.float32)
                if temperature > 0:
                    modified_image[:, :, 0] *= 1 + temperature
                    modified_image[:, :, 1] *= 1 + temperature * 0.4
                elif temperature < 0:
                    modified_image[:, :, 0] *= 1 + temperature * 0.2
                    modified_image[:, :, 2] *= 1 - temperature

                modified_image = np.clip(modified_image, 0, 255)
                modified_image = modified_image.astype(np.uint8)
                modified_image = modified_image / 255
                image = torch.from_numpy(modified_image).unsqueeze(0)

            # apply HSV, [takes in tensor][gives out PIL]
            _h, _s, _v = tensor2pil(image).convert('HSV').split()
            if hue != 0 :
                log_to_console("in hue")
                _h = image_hue_offset(_h, hue)
            if saturation != 0 :
                log_to_console("in saturation")
                _s = image_gray_offset(_s, saturation)
            if value != 0 :
                log_to_console("in value")
                _v = image_gray_offset(_v, value)
            return_image = image_channel_merge((_h, _s, _v), 'HSV')

           # apply color balance [takes in PIL][gives out PIL]
            return_image = color_balance(return_image,
                    [cyan_red, magenta_green, yellow_blue],
                    [cyan_red, magenta_green, yellow_blue],
                    [cyan_red, magenta_green, yellow_blue],
                            shadow_center=0.15,
                            midtone_center=0.5,
                            midtone_max=1,
                            preserve_luminosity=True)

            #apply gamma [takes in tensor][gives out PIL image]
            if type(return_image) == Image.Image:
                log_to_console("gamma trigger")
                return_image = gamma_trans(return_image, gamma)
            else:
                return_image = gamma_trans(tensor2pil(return_image), gamma)

            #apply contrast [takes and gives out PIL]
            if contrast != 1:
                log_to_console("in contrast")
                contrast_image = ImageEnhance.Contrast(return_image)
                return_image = contrast_image.enhance(factor=contrast)

            # apply exposure [takes in tensor][gives out PIL]
            if exposure:
                log_to_console("in exposure")
                return_image = pil2tensor(return_image)
                temp = return_image.detach().clone().cpu().numpy().astype(np.float32)
                more = temp[:, :, :, :3] > 0
                temp[:, :, :, :3][more] *= pow(2, exposure / 32)
                if exposure < 0:
                    bp = -exposure / 250
                    scale = 1 / (1 - bp)
                    temp = np.clip((temp - bp) * scale, 0.0, 1.0)
                return_image = tensor2pil(torch.from_numpy(temp))


            return_image = chop_image_v2(original_image, return_image, blend_mode="normal", opacity=100)
            return_image.paste(original_image, mask=ImageChops.invert(mask))
            if original_image.mode == 'RGBA':
                return_image = RGB2RGBA(return_image, original_image.split()[-1])

            return_images.append(pil2tensor(return_image))

        return (torch.cat(return_images, dim=0),)

NODE_CLASS_MAPPINGS = {"ColorCorrect" : ColorCorrect,}
NODE_DISPLAY_NAME_MAPPINGS = {"ColorCorrect": PACK_NAME + " ColorCorrect",}

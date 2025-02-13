from comfy import model_management
from .image_utils import *
from .utils import *

import comfy.utils

# from official comfy implementation
def composite_masked(destination, source, x, y, mask = None, multiplier = 8, resize_source = False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[3], top + source.shape[2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination

class CropComposite:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "concat_image" : ("IMAGE",),
                "crop_data" : ("crop_data", ),
                "grow": ("INT", {"default": 4, "min": -999, "max": 999, "step": 1}),
                "blur": ("INT", {"default": 4, "min": 0, "max": 999, "step": 1}),
            },
            "optional" : {
                "upscale_model" : ("UPSCALE_MODEL",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = PACK_NAME

    def execute(self, concat_image:torch.Tensor, grow:int, blur:int, crop_data:dict, upscale_model=None):
        
        # crop the generated image from the concatenated image
        crop_x = min(crop_data["crop_x"], concat_image.shape[2] - 1)
        crop_y = min(0, concat_image.shape[1] - 1)
        to_crop_x = crop_data["crop_width"] + crop_x
        to_crop_y = crop_data["crop_height"] + crop_y
        cropped_image = concat_image[:, crop_y:to_crop_y, crop_x:to_crop_x, :]
        
        log_to_console(f"""Cropped image from concatenated image""")

        # upscale the cropped image
        # fist upscale using scale
        samples = cropped_image.movedim(-1, 1)
        upscale_width = round(samples.shape[3] * crop_data["scale"])
        upscale_height = round(samples.shape[2] * crop_data["scale"])
        upscaled_image = comfy.utils.common_upscale(samples, upscale_width, upscale_height, "lanczos", "disabled")
        upscaled_image = upscaled_image.movedim(1, -1)
        log_to_console(f"upscaled using scale")

        # OPTIONAL
        # second upscale using model
        if upscale_model is not None:
            device = model_management.get_torch_device()
            memory_required = model_management.module_size(upscale_model.model)
            memory_required += (512 * 512 * 3) * upscaled_image.element_size() * max(upscale_model.scale, 1.0) * 384.0 
            memory_required += upscaled_image.nelement() * upscaled_image.element_size()
            model_management.free_memory(memory_required, device)

            upscale_model.to(device)
            in_img = upscaled_image.movedim(-1, -3).to(device)

            tile = 512
            overlap = 32
            oom = True
            while oom:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = comfy.utils.ProgressBar(steps)
                    upscaled_image = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                    oom = False
                except model_management.OOM_EXCEPTION as e:
                    tile //= 2
                    if tile < 128:
                        raise e
                upscale_model.to("cpu")

                upscaled_image = torch.clamp(upscaled_image.movedim(-3, -1), min=0, max=1.0)
                log_to_console(f"""upscaled image using model""")
        
        # grow and blur mask
        l_masks = []
        ret_masks = []
        mask = crop_data["destn_mask"]
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)

        for m in mask:
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_masks)):

            _mask = l_masks[i]
            ret_masks.append(expand_mask(image2mask(_mask), grow, blur))

        destn_mask = torch.cat(ret_masks, dim=0)
        log_to_console(f"mask optimized, grow = {grow}, blur = {blur}")

        # image compositing
        destn_img = crop_data["destn_img"].clone().movedim(-1, 1)
        out_image:torch.Tensor = composite_masked(destn_img, upscaled_image.movedim(-1, 1), crop_data["composite_x"], crop_data["composite_y"], destn_mask, 1).movedim(1, -1)
        log_to_console(f"""Image composited""")
        return (out_image,)
    
NODE_CLASS_MAPPINGS = {"CropComposite" : CropComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"CropComposite" : PACK_NAME + " CropComposite"}
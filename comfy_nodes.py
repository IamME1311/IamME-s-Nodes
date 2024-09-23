import comfy
import torch

MAX_RESOLUTION = 16384

class AspectEmptyLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "aspect_width_ratio" : ("INT",{"default":9, "min":0}),
                "aspect_height_ratio" : ("INT",{"default":16, "min":0}),
                "width_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "aspect_latent_gen"

    CATEGORY = "IamME"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def aspect_latent_gen(self, aspect_width_ratio, aspect_height_ratio, width_override, batch_size=1):
        if width_override > 0:
            width = width_override
            height = int((aspect_height_ratio * width) / aspect_width_ratio)
        else:
            base_size = 1024
            if aspect_width_ratio <= aspect_height_ratio:
                width = base_size
                height = int((aspect_height_ratio * width) / aspect_width_ratio)
            else:
                height = base_size
                width = int((aspect_width_ratio * height) / aspect_height_ratio)
    
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, )





















NODE_CLASS_MAPPINGS = {
    "AspectLatentImage" : AspectEmptyLatentImage
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectLatentImage" : "Aspect Empty Latent Image"
}
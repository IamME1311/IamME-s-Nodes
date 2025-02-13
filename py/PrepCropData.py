from .image_utils import *
from .utils import *

class PrepCropData:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "crop_width" : ("INT", {"default":512, "step":8, "forceInput":True,},),
                "crop_height" : ("INT", {"default":512, "step":8, "forceInput":True,},),
                "crop_x" : ("INT", {"default":0, "step":1, "forceInput":True,}),
                "composite_x" : ("INT", {"default":0, "step":1, "forceInput":True,}),
                "composite_y" : ("INT", {"default":0, "step":1, "forceInput":True,}),
                "scale" : ("FLOAT", {"default":0, "step":1, "forceInput":True,}),
                "destn_img" : ("IMAGE",),
                "destn_mask" : ("MASK",),
            }
        }
    
    FUNCTION = "execute"
    RETURN_TYPES = ("crop_data",)
    CATEGORY = PACK_NAME


    def execute(self, 
                crop_width:int, 
                crop_height:int,
                crop_x:int,
                composite_x:int,
                composite_y:int,
                scale:float,
                destn_img:torch.Tensor,
                destn_mask:torch.Tensor):
        
        crop_data = {
            "crop_width" : crop_width, 
            "crop_height": crop_height,
            "crop_x" : crop_x,
            "composite_x" : composite_x,
            "composite_y" : composite_y,
            "scale": scale,
            "destn_img" : destn_img,
            "destn_mask" : destn_mask          
        }
        return (crop_data,)
    

NODE_CLASS_MAPPINGS = {"PrepCropData" : PrepCropData}
NODE_DISPLAY_NAME_MAPPINGS = {"PrepCropData" : PACK_NAME + " PrepCropData"}
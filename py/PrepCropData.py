from .image_utils import *
from .utils import *

class PrepCropData:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "target_width" : ("INT", {"default":512, "step":8, "forceInput":True,},),
                "target_height" : ("INT", {"default":512, "step":8, "forceInput":True,},),
                "target_x" : ("INT", {"default":0, "step":1, "forceInput":True,}),
                "x_offset_of_ori" : ("INT", {"default":0, "step":1, "forceInput":True,}),
                "y_offset_of_ori" : ("INT", {"default":0, "step":1, "forceInput":True,}),
                "scale" : ("FLOAT", {"default":0, "step":1, "forceInput":True,}),
                "destn_img" : ("IMAGE",),
                "destn_mask" : ("MASK",),
            }
        }
    
    FUNCTION = "execute"
    RETURN_TYPES = ("crop_data",)
    CATEGORY = PACK_NAME


    def execute(self, 
                target_width:int, 
                target_height:int,
                target_x:int,
                x_offset_of_ori:int,
                y_offset_of_ori:int,
                scale:float,
                destn_img:torch.Tensor,
                destn_mask:torch.Tensor):
        
        crop_data = {
            "crop_width" : target_width, 
            "crop_height": target_height,
            "crop_x" : target_x,
            "composite_x" : x_offset_of_ori,
            "composite_y" : y_offset_of_ori,
            "scale": scale,
            "destn_img" : destn_img,
            "destn_mask" : destn_mask          
        }
        return (crop_data,)
    

NODE_CLASS_MAPPINGS = {"PrepCropData" : PrepCropData}
NODE_DISPLAY_NAME_MAPPINGS = {"PrepCropData" : PACK_NAME + " PrepCropData"}
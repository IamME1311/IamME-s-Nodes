import torch
from .utils import *
import torch.nn.functional as F

class ImageConcatenate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            # For Sri :  do interpolation here
            "method": ("BOOLEAN", {"default": True, "label_on" : "match size", "label_off" : "pad",}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "WIDTH_OUT", "HEIGHT_OUT", "X_OFF",)
    FUNCTION = "execute"
    CATEGORY = PACK_NAME
    DESCRIPTION = """
Concatenates the second image to the first image in the specified direction. Optionally resizes the second image to match the dimensions of the first.
    """

    def resize_image(self, image:torch.Tensor, target_width:int, target_height:int) -> torch.Tensor:
        # Resize the image using PyTorch's interpolation function
        return F.interpolate(image, size=(target_height, target_width), mode="bilinear", align_corners=False)

    # add interpolation variable here
    def execute(self, image1:torch.Tensor, image2:torch.Tensor, method:bool,) -> tuple:
        """this is the docstring"""
        # Step 1: Handle batch size mismatch by repeating smaller batch images to match the largest batch size
        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]
        

        if batch_size1 != batch_size2:
            max_batch_size = max(batch_size1, batch_size2)
            image1 = image1.repeat(max_batch_size // batch_size1, 1, 1, 1)
            image2 = image2.repeat(max_batch_size // batch_size2, 1, 1, 1)

        # Use the shape of the first image as the target
        target_shape = image1.shape
        original_width = image2.shape[2]
        original_height = image2.shape[1]
        aspect_ratio = original_width / original_height # for keeping the proportion

        target_height = target_shape[1]
        target_width = int(target_height*aspect_ratio)
        # Step 2: Resize the second image to match the first image's dimensions if required
        if method:
            # Resize image2 using the resize_image function
            image2_resized = self.resize_image(
                image2.movedim(-1, 1),  # Convert to (B, C, H, W) format
                target_width,
                target_height
            ).movedim(1, -1)  # Convert back to (B, H, W, C) format
        else:
            ########################################
            #               WIP                    #
            ########################################
            pad_top = pad_left = pad_right = pad_bottom = 0
            ratio = min(target_width / original_width, target_height / original_height)
            new_width = round(original_width*ratio)
            new_height = round(original_height*ratio)
            
            pad_left = (target_width - new_width) // 2
            pad_right = target_width - new_width - pad_left
            pad_top = (target_height - new_height) // 2
            pad_bottom = target_height - new_height - pad_top

            outputs = image2.permute(0,3,1,2)
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)
            image2_resized = outputs.permute(0,2,3,1)

        # Step 3: Concatenate the images on left side
        concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width with image2 on the left
        
        width = concatenated_image.shape[2] - image2_resized.shape[2] # with of cropped image
        height = concatenated_image.shape[1] # height of second image
        x_off = image2_resized.shape[2] # width of first image

        # Step 4: Return the concatenated image
        return (concatenated_image, width, height, x_off,)

NODE_CLASS_MAPPINGS = {
    "ImageConcatenate": ImageConcatenate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageConcatenate": PACK_NAME + " Image Concat"
}